#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <csignal>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <sys/wait.h>
#include <sys/resource.h>
#include <opencv2/highgui.hpp>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <xir/attrs/attrs.hpp>
#include <xir/graph/graph.hpp>
#include <xir/graph/subgraph.hpp>
#include <xir/tensor/tensor.hpp>

#include <vart/runner_ext.hpp>
#include <vart/tensor_buffer.hpp>

static volatile std::sig_atomic_t
    g_stop_requested = 0;

static void handle_stop_signal(int)
{
    g_stop_requested = 1;
}

constexpr int NUM_CLASSES = 80;
constexpr int ANCHORS_PER_SCALE = 3;

constexpr float CONF_THRESHOLD = 0.50f;
constexpr float NMS_THRESHOLD  = 0.65f;

constexpr float MODEL_SCALE = 0.00392156f;

static const std::array<const char*, NUM_CLASSES> COCO_CLASSES = {{
    "person",        "bicycle",       "car",           "motorbike",
    "aeroplane",     "bus",           "train",         "truck",
    "boat",          "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench",         "bird",          "cat",
    "dog",           "horse",         "sheep",         "cow",
    "elephant",      "bear",          "zebra",         "giraffe",
    "backpack",      "umbrella",      "handbag",       "tie",
    "suitcase",      "frisbee",       "skis",          "snowboard",
    "sports ball",   "kite",          "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",           "fork",          "knife",
    "spoon",         "bowl",          "banana",        "apple",
    "sandwich",      "orange",        "broccoli",      "carrot",
    "hot dog",       "pizza",         "donut",         "cake",
    "chair",         "sofa",          "pottedplant",   "bed",
    "diningtable",   "toilet",        "tvmonitor",     "laptop",
    "mouse",         "remote",        "keyboard",      "cell phone",
    "microwave",     "oven",          "toaster",       "sink",
    "refrigerator",  "book",          "clock",         "vase",
    "scissors",      "teddy bear",    "hair drier",    "toothbrush"
}};

struct YoloBox {
    float cx;
    float cy;
    float width;
    float height;
    float objectness;

    std::array<float, NUM_CLASSES> class_scores;
};

struct Detection {
    int class_id;
    float score;

    float x;
    float y;
    float width;
    float height;
};

static float sigmoid(float value)
{
    return 1.0f / (1.0f + std::exp(-value));
}

static std::vector<int> zero_index(const xir::Tensor* tensor)
{
    return std::vector<int>(tensor->get_shape().size(), 0);
}

static void print_shape(const xir::Tensor* tensor)
{
    const auto shape = tensor->get_shape();

    std::cout << "[";

    for (std::size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];

        if (i + 1 < shape.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]";
}

static float intersection_over_union(
    const YoloBox& a,
    const YoloBox& b)
{
    const float left =
        std::max(
            a.cx - a.width / 2.0f,
            b.cx - b.width / 2.0f);

    const float right =
        std::min(
            a.cx + a.width / 2.0f,
            b.cx + b.width / 2.0f);

    const float top =
        std::max(
            a.cy - a.height / 2.0f,
            b.cy - b.height / 2.0f);

    const float bottom =
        std::min(
            a.cy + a.height / 2.0f,
            b.cy + b.height / 2.0f);

    const float overlap_width = right - left;
    const float overlap_height = bottom - top;

    if (overlap_width < 0.0f ||
        overlap_height < 0.0f) {
        return 0.0f;
    }

    const float intersection =
        overlap_width * overlap_height;

    const float union_area =
        a.width * a.height +
        b.width * b.height -
        intersection;

    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return intersection / union_area;
}

static const std::array<std::array<float, 2>, 3>&
anchors_for_feature_map(int feature_height)
{
    /*
     * Exact anchors from the YOLOv5 Nano prototxt.
     */

    static const std::array<std::array<float, 2>, 3> anchors_80 = {{
        {{10.0f, 13.0f}},
        {{16.0f, 30.0f}},
        {{33.0f, 23.0f}}
    }};

    static const std::array<std::array<float, 2>, 3> anchors_40 = {{
        {{30.0f, 61.0f}},
        {{62.0f, 45.0f}},
        {{59.0f, 119.0f}}
    }};

    static const std::array<std::array<float, 2>, 3> anchors_20 = {{
        {{116.0f, 90.0f}},
        {{156.0f, 198.0f}},
        {{373.0f, 326.0f}}
    }};

    if (feature_height == 80) {
        return anchors_80;
    }

    if (feature_height == 40) {
        return anchors_40;
    }

    return anchors_20;
}

static bool decode_output_tensor(
    vart::TensorBuffer* output,
    int network_width,
    int network_height,
    std::vector<YoloBox>& boxes)
{
    auto* tensor = output->get_tensor();

    const auto shape = tensor->get_shape();

    if (shape.size() != 4) {
        std::cerr
            << "ERROR: YOLO output is not NHWC\n";
        return false;
    }

    const int batch = shape[0];
    const int height = shape[1];
    const int width = shape[2];
    const int channels = shape[3];

    if (batch != 1) {
        std::cerr
            << "ERROR: Only batch size 1 is supported\n";
        return false;
    }

    const int values_per_anchor =
        5 + NUM_CLASSES;

    const int expected_channels =
        ANCHORS_PER_SCALE * values_per_anchor;

    if (channels != expected_channels) {
        std::cerr
            << "ERROR: Expected "
            << expected_channels
            << " channels but tensor has "
            << channels
            << "\n";
        return false;
    }

    if (height != 80 &&
        height != 40 &&
        height != 20) {

        std::cerr
            << "ERROR: Unexpected YOLO feature-map size: "
            << height << "x" << width
            << "\n";

        return false;
    }

    if (!tensor->has_attr("fix_point")) {
        std::cerr
            << "ERROR: Output tensor has no fix_point\n";
        return false;
    }

    const int fix_point =
        tensor->get_attr<int>("fix_point");

    /*
     * Vitis AI tensor_scale(output) for XINT8:
     *
     * scale = 2^(-fix_point)
     */
    const float dequant_scale =
        std::ldexp(1.0f, -fix_point);

    uint64_t address = 0;
    std::size_t size = 0;

    std::tie(address, size) =
        output->data(zero_index(tensor));

    if (address == 0 || size == 0) {
        std::cerr
            << "ERROR: Cannot access YOLO output buffer\n";
        return false;
    }

    const auto* result =
        reinterpret_cast<const int8_t*>(address);

    const auto& anchors =
        anchors_for_feature_map(height);

    /*
     * Vitis AI source:
     *
     * conf_desigmoid =
     *   -log(1 / conf_threshold - 1)
     */
    const float conf_desigmoid =
        -std::log(
            1.0f / CONF_THRESHOLD - 1.0f);

    std::cout
        << "\nDecoding "
        << height << "x" << width
        << " tensor"
        << " FIX_POINT=" << fix_point
        << " SCALE=" << dequant_scale
        << "\n";

    for (int h = 0; h < height; ++h) {

        for (int w = 0; w < width; ++w) {

            for (int anchor = 0;
                 anchor < ANCHORS_PER_SCALE;
                 ++anchor) {

                const int index =
                    ((h * width + w) *
                     ANCHORS_PER_SCALE +
                     anchor) *
                    values_per_anchor;

                const float object_logit =
                    static_cast<float>(
                        result[index + 4]) *
                    dequant_scale;

                /*
                 * Same early filtering used by
                 * Vitis AI yolov3.cpp.
                 */
                if (object_logit <
                    conf_desigmoid) {
                    continue;
                }

                YoloBox box{};

                const float tx =
                    static_cast<float>(
                        result[index + 0]) *
                    dequant_scale;

                const float ty =
                    static_cast<float>(
                        result[index + 1]) *
                    dequant_scale;

                const float tw =
                    static_cast<float>(
                        result[index + 2]) *
                    dequant_scale;

                const float th =
                    static_cast<float>(
                        result[index + 3]) *
                    dequant_scale;

                /*
                 * Exact YOLOv5 equations from
                 * Vitis AI 3.0 yolov3.cpp.
                 */

                box.cx =
                    (sigmoid(tx) * 2.0f -
                     0.5f +
                     static_cast<float>(w)) /
                    static_cast<float>(width);

                box.cy =
                    (sigmoid(ty) * 2.0f -
                     0.5f +
                     static_cast<float>(h)) /
                    static_cast<float>(height);

                const float sw =
                    sigmoid(tw) * 2.0f;

                const float sh =
                    sigmoid(th) * 2.0f;

                box.width =
                    std::pow(sw, 2.0f) *
                    anchors[anchor][0] /
                    static_cast<float>(
                        network_width);

                box.height =
                    std::pow(sh, 2.0f) *
                    anchors[anchor][1] /
                    static_cast<float>(
                        network_height);

                box.objectness =
                    sigmoid(object_logit);

                for (int class_id = 0;
                     class_id < NUM_CLASSES;
                     ++class_id) {

                    const float class_logit =
                        static_cast<float>(
                            result[
                                index +
                                5 +
                                class_id]) *
                        dequant_scale;

                    box.class_scores[class_id] =
                        box.objectness *
                        sigmoid(class_logit);
                }

                boxes.push_back(box);
            }
        }
    }

    return true;
}

static std::vector<Detection>
apply_vitis_ai_nms(
    const std::vector<YoloBox>& boxes)
{
    std::vector<Detection> detections;

    /*
     * Match Vitis AI:
     *
     * NMS is performed independently
     * for every class.
     */
    for (int class_id = 0;
         class_id < NUM_CLASSES;
         ++class_id) {

        std::vector<std::pair<float, std::size_t>>
            order;

        order.reserve(boxes.size());

        for (std::size_t i = 0;
             i < boxes.size();
             ++i) {

            order.push_back({
                boxes[i].class_scores[class_id],
                i
            });
        }

        /*
         * Vitis AI default applyNMS()
         * uses std::sort, not stable_sort.
         */
        std::sort(
            order.begin(),
            order.end(),
            [](const auto& a,
               const auto& b) {
                return a.first > b.first;
            });

        std::vector<bool>
            exists(boxes.size(), true);

        for (std::size_t oi = 0;
             oi < order.size();
             ++oi) {

            const std::size_t i =
                order[oi].second;

            if (!exists[i]) {
                continue;
            }

            const float score =
                boxes[i].class_scores[class_id];

            if (score < CONF_THRESHOLD) {
                exists[i] = false;
                continue;
            }

            Detection detection{};

            detection.class_id =
                class_id;

            detection.score =
                score;

            detection.x =
                boxes[i].cx -
                boxes[i].width / 2.0f;

            detection.y =
                boxes[i].cy -
                boxes[i].height / 2.0f;

            detection.width =
                boxes[i].width;

            detection.height =
                boxes[i].height;

            detections.push_back(
                detection);

            for (std::size_t oj = oi + 1;
                 oj < order.size();
                 ++oj) {

                const std::size_t j =
                    order[oj].second;

                if (!exists[j]) {
                    continue;
                }

                const float iou =
                    intersection_over_union(
                        boxes[j],
                        boxes[i]);

                /*
                 * Exact Vitis AI comparison:
                 *
                 * if (ovr >= nms)
                 */
                if (iou >= NMS_THRESHOLD) {
                    exists[j] = false;
                }
            }
        }
    }

    return detections;
}


/* ========================================================= */
/* KV260 HLS Sobel userspace interface                       */
/* ========================================================= */

struct SobelRunRequest {
    std::uint32_t width;
    std::uint32_t height;
};

#define SOBEL_IOCTL_RUN \
    _IOW('S', 0x01, SobelRunRequest)

static constexpr std::size_t SOBEL_MAX_WIDTH =
    1920;

static constexpr std::size_t SOBEL_MAX_HEIGHT =
    1080;

static constexpr std::size_t SOBEL_MAX_FRAME_BYTES =
    SOBEL_MAX_WIDTH * SOBEL_MAX_HEIGHT;

class SobelHardwareContext
{
public:
    SobelHardwareContext() = default;

    ~SobelHardwareContext()
    {
        shutdown();
    }

    SobelHardwareContext(
        const SobelHardwareContext&) = delete;

    SobelHardwareContext& operator=(
        const SobelHardwareContext&) = delete;

    bool initialize()
    {
        if (initialized_) {
            return true;
        }

        const long page_size_raw =
            ::sysconf(_SC_PAGESIZE);

        if (page_size_raw <= 0) {
            std::cerr
                << "SOBEL ERROR: cannot determine page size\n";
            return false;
        }

        const std::size_t page_size =
            static_cast<std::size_t>(
                page_size_raw);

        buffer_stride_ =
            ((SOBEL_MAX_FRAME_BYTES +
              page_size - 1) /
             page_size) *
            page_size;

        total_dma_bytes_ =
            2U * buffer_stride_;

        fd_ =
            ::open(
                "/dev/sobel_accel",
                O_RDWR);

        if (fd_ < 0) {
            std::cerr
                << "SOBEL ERROR: open(/dev/sobel_accel): "
                << std::strerror(errno)
                << "\n";

            return false;
        }

        mapped_ =
            ::mmap(
                nullptr,
                total_dma_bytes_,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd_,
                0);

        if (mapped_ == MAP_FAILED) {
            std::cerr
                << "SOBEL ERROR: mmap(): "
                << std::strerror(errno)
                << "\n";

            mapped_ = nullptr;

            ::close(fd_);
            fd_ = -1;

            return false;
        }

        input_dma_ =
            static_cast<std::uint8_t*>(
                mapped_);

        output_dma_ =
            input_dma_ + buffer_stride_;

        initialized_ = true;

        std::cout
            << "Sobel hardware initialized once\n"
            << "DMA map = "
            << total_dma_bytes_
            << " bytes\n";

        return true;
    }

    bool process(
        const cv::Mat& image,
        cv::Mat& sobel_output,
        double& sobel_hw_ms)
    {
        if (image.empty()) {
            std::cerr
                << "SOBEL ERROR: empty input image\n";
            return false;
        }

        if (image.cols <= 0 ||
            image.rows <= 0 ||
            static_cast<std::size_t>(
                image.cols) >
                SOBEL_MAX_WIDTH ||
            static_cast<std::size_t>(
                image.rows) >
                SOBEL_MAX_HEIGHT) {

            std::cerr
                << "SOBEL ERROR: image "
                << image.cols
                << "x"
                << image.rows
                << " exceeds maximum "
                << SOBEL_MAX_WIDTH
                << "x"
                << SOBEL_MAX_HEIGHT
                << "\n";

            return false;
        }

        if (!initialize()) {
            return false;
        }

        const std::size_t frame_bytes =
            static_cast<std::size_t>(
                image.cols) *
            static_cast<std::size_t>(
                image.rows);

        cv::Mat gray;

        cv::cvtColor(
            image,
            gray,
            cv::COLOR_BGR2GRAY);

        for (int row = 0;
             row < gray.rows;
             ++row) {

            std::memcpy(
                input_dma_ +
                    static_cast<std::size_t>(
                        row) *
                    static_cast<std::size_t>(
                        gray.cols),

                gray.ptr<std::uint8_t>(row),

                static_cast<std::size_t>(
                    gray.cols));
        }

        std::memset(
            output_dma_,
            0,
            frame_bytes);

        SobelRunRequest request{};

        request.width =
            static_cast<std::uint32_t>(
                image.cols);

        request.height =
            static_cast<std::uint32_t>(
                image.rows);

        const auto sobel_hw_start =
            std::chrono::steady_clock::now();

        const int ioctl_result =
            ::ioctl(
                fd_,
                SOBEL_IOCTL_RUN,
                &request);

        const auto sobel_hw_end =
            std::chrono::steady_clock::now();

        sobel_hw_ms =
            std::chrono::duration<
                double,
                std::milli>(
                    sobel_hw_end -
                    sobel_hw_start)
                .count();

        if (ioctl_result < 0) {
            std::cerr
                << "SOBEL ERROR: ioctl(SOBEL_RUN): "
                << std::strerror(errno)
                << "\n";

            return false;
        }

        cv::Mat hardware_view(
            image.rows,
            image.cols,
            CV_8UC1,
            output_dma_,
            static_cast<std::size_t>(
                image.cols));

        /*
         * Clone because the same DMA output buffer
         * will be reused by the next video frame.
         */
        sobel_output =
            hardware_view.clone();

        return true;
    }

private:
    void shutdown()
    {
        if (mapped_ != nullptr) {
            ::munmap(
                mapped_,
                total_dma_bytes_);

            mapped_ = nullptr;
        }

        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }

        input_dma_ = nullptr;
        output_dma_ = nullptr;
        initialized_ = false;
    }

    int fd_ = -1;

    void* mapped_ = nullptr;

    std::uint8_t* input_dma_ = nullptr;
    std::uint8_t* output_dma_ = nullptr;

    std::size_t buffer_stride_ = 0;
    std::size_t total_dma_bytes_ = 0;

    bool initialized_ = false;
};

/* ========================================================= */


/* ========================================================= */
/* Minimal HTTP/MJPEG live-stream server                      */
/* ========================================================= */

class MjpegHttpServer
{
public:
    MjpegHttpServer() = default;

    ~MjpegHttpServer()
    {
        shutdown();
    }

    MjpegHttpServer(
        const MjpegHttpServer&) = delete;

    MjpegHttpServer& operator=(
        const MjpegHttpServer&) = delete;

    bool start(int port)
    {
        if (port <= 0 ||
            port > 65535) {

            std::cerr
                << "STREAM ERROR: invalid port "
                << port
                << "\n";

            return false;
        }

        server_fd_ =
            ::socket(
                AF_INET,
                SOCK_STREAM,
                0);

        if (server_fd_ < 0) {
            std::cerr
                << "STREAM ERROR: socket(): "
                << std::strerror(errno)
                << "\n";

            return false;
        }

        int reuse = 1;

        ::setsockopt(
            server_fd_,
            SOL_SOCKET,
            SO_REUSEADDR,
            &reuse,
            sizeof(reuse));

        sockaddr_in address{};

        address.sin_family =
            AF_INET;

        address.sin_addr.s_addr =
            htonl(INADDR_ANY);

        address.sin_port =
            htons(
                static_cast<std::uint16_t>(
                    port));

        if (::bind(
                server_fd_,
                reinterpret_cast<sockaddr*>(
                    &address),
                sizeof(address)) < 0) {

            std::cerr
                << "STREAM ERROR: bind(): "
                << std::strerror(errno)
                << "\n";

            shutdown();
            return false;
        }

        if (::listen(
                server_fd_,
                1) < 0) {

            std::cerr
                << "STREAM ERROR: listen(): "
                << std::strerror(errno)
                << "\n";

            shutdown();
            return false;
        }

        const int flags =
            ::fcntl(
                server_fd_,
                F_GETFL,
                0);

        if (flags >= 0) {
            ::fcntl(
                server_fd_,
                F_SETFL,
                flags | O_NONBLOCK);
        }

        port_ = port;

        std::cout
            << "MJPEG live stream enabled\n"
            << "Listening on TCP port "
            << port_
            << "\n"
            << "Open in laptop browser:\n"
            << "  http://<KV260-IP>:"
            << port_
            << "/\n";

        return true;
    }

    void publish(
        const cv::Mat& frame)
    {
        if (server_fd_ < 0 ||
            frame.empty()) {

            return;
        }

        /*
         * Do not block the perception loop waiting
         * for a browser to connect.
         */
        if (client_fd_ < 0) {

            sockaddr_in client_address{};
            socklen_t client_length =
                sizeof(client_address);

            const int accepted =
                ::accept(
                    server_fd_,
                    reinterpret_cast<sockaddr*>(
                        &client_address),
                    &client_length);

            if (accepted < 0) {

                if (errno == EAGAIN ||
                    errno == EWOULDBLOCK ||
                    errno == EINTR) {

                    return;
                }

                return;
            }

            client_fd_ =
                accepted;

            const std::string response =
                "HTTP/1.1 200 OK\r\n"
                "Cache-Control: no-cache\r\n"
                "Pragma: no-cache\r\n"
                "Connection: close\r\n"
                "Content-Type: multipart/x-mixed-replace; "
                "boundary=frame\r\n"
                "\r\n";

            if (!send_all(
                    reinterpret_cast<
                        const std::uint8_t*>(
                            response.data()),
                    response.size())) {

                close_client();
                return;
            }

            std::cout
                << "MJPEG browser client connected\n";
        }

        std::vector<std::uint8_t>
            jpeg_buffer;

        const std::vector<int>
            jpeg_parameters = {
                cv::IMWRITE_JPEG_QUALITY,
                80
            };

        if (!cv::imencode(
                ".jpg",
                frame,
                jpeg_buffer,
                jpeg_parameters)) {

            return;
        }

        const std::string part_header =
            "--frame\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: " +
            std::to_string(
                jpeg_buffer.size()) +
            "\r\n\r\n";

        if (!send_all(
                reinterpret_cast<
                    const std::uint8_t*>(
                        part_header.data()),
                part_header.size()) ||

            !send_all(
                jpeg_buffer.data(),
                jpeg_buffer.size())) {

            close_client();
            return;
        }

        static const std::uint8_t
            trailer[] = {
                '\r', '\n'
            };

        if (!send_all(
                trailer,
                sizeof(trailer))) {

            close_client();
        }
    }

private:
    bool send_all(
        const std::uint8_t* data,
        std::size_t size)
    {
        std::size_t sent = 0;

        while (sent < size) {

            const ssize_t result =
                ::send(
                    client_fd_,
                    data + sent,
                    size - sent,
                    MSG_NOSIGNAL);

            if (result > 0) {
                sent +=
                    static_cast<std::size_t>(
                        result);

                continue;
            }

            if (result < 0 &&
                errno == EINTR) {

                continue;
            }

            return false;
        }

        return true;
    }

    void close_client()
    {
        if (client_fd_ >= 0) {
            ::close(client_fd_);
            client_fd_ = -1;

            std::cout
                << "MJPEG browser client disconnected\n";
        }
    }

    void shutdown()
    {
        close_client();

        if (server_fd_ >= 0) {
            ::close(server_fd_);
            server_fd_ = -1;
        }
    }

    int server_fd_ = -1;
    int client_fd_ = -1;
    int port_ = 0;
};

/* ========================================================= */


/* ========================================================= */
/* ARM CPU Sobel reference                                   */
/* Matches the HLS Sobel arithmetic                          */
/* ========================================================= */

static bool run_sobel_cpu_reference(
    const cv::Mat& image,
    const std::string& output_path,
    double& cpu_sobel_ms)
{
    if (image.empty()) {
        std::cerr
            << "CPU SOBEL ERROR: empty image\n";
        return false;
    }

    cv::Mat gray;

    cv::cvtColor(
        image,
        gray,
        cv::COLOR_BGR2GRAY);

    cv::Mat output =
        cv::Mat::zeros(
            gray.rows,
            gray.cols,
            CV_8UC1);

    /*
     * Time only the Sobel computation.
     *
     * Grayscale conversion and image saving are intentionally
     * excluded so the number can later be compared with the
     * HLS hardware execution time.
     */
    const auto cpu_start =
        std::chrono::steady_clock::now();

    for (int y = 1;
         y < gray.rows - 1;
         ++y) {

        const std::uint8_t* row0 =
            gray.ptr<std::uint8_t>(y - 1);

        const std::uint8_t* row1 =
            gray.ptr<std::uint8_t>(y);

        const std::uint8_t* row2 =
            gray.ptr<std::uint8_t>(y + 1);

        std::uint8_t* dst =
            output.ptr<std::uint8_t>(y);

        for (int x = 1;
             x < gray.cols - 1;
             ++x) {

            const int p00 = row0[x - 1];
            const int p01 = row0[x];
            const int p02 = row0[x + 1];

            const int p10 = row1[x - 1];
            const int p12 = row1[x + 1];

            const int p20 = row2[x - 1];
            const int p21 = row2[x];
            const int p22 = row2[x + 1];

            const int gx =
                -p00 + p02
                -2 * p10 + 2 * p12
                -p20 + p22;

            const int gy =
                -p00 - 2 * p01 - p02
                +p20 + 2 * p21 + p22;

            int magnitude =
                std::abs(gx) +
                std::abs(gy);

            if (magnitude > 255)
                magnitude = 255;

            dst[x] =
                static_cast<std::uint8_t>(
                    magnitude);
        }
    }

    const auto cpu_end =
        std::chrono::steady_clock::now();

    cpu_sobel_ms =
        std::chrono::duration<double, std::milli>(
            cpu_end -
            cpu_start).count();

    std::cout
        << "\nCPU SOBEL REFERENCE\n"
        << "CPU Sobel execution = "
        << cpu_sobel_ms
        << " ms\n";

    if (!cv::imwrite(
            output_path,
            output)) {

        std::cerr
            << "CPU SOBEL ERROR: cannot save "
            << output_path
            << "\n";

        return false;
    }

    std::cout
        << "CPU Sobel output saved = "
        << output_path
        << "\n";

    return true;
}

/* ========================================================= */


// Presentation uses separate branches: original color -> YOLO;
// actual HLS Sobel edges -> ARM/OpenCV lane candidate postprocessing.
static constexpr int PANEL_W = 624;
static constexpr int PANEL_H = 352;
static const cv::Size PRESENTATION_SIZE(1280, 520);

static void caption(cv::Mat& image, const std::string& text, int x, int y,
                    double scale = 0.55, cv::Scalar color = cv::Scalar(235, 240, 245))
{
    cv::putText(image, text, {x, y}, cv::FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv::LINE_AA);
}

// Fit each source without distorting its aspect ratio; draw directly into the composite.
static cv::Rect fitted_rect(const cv::Size& size, int x)
{
    const double scale = std::min(double(PANEL_W) / size.width,
                                  double(PANEL_H) / size.height);
    const int w = std::max(1, cvRound(size.width * scale));
    const int h = std::max(1, cvRound(size.height * scale));
    return {x + (PANEL_W - w) / 2, 100 + (PANEL_H - h) / 2, w, h};
}

static void lane_view(const cv::Mat& sobel, cv::Mat panel)
{
    cv::Mat edges;
    cv::resize(sobel, edges, panel.size(), 0, 0, cv::INTER_AREA);
    cv::cvtColor(edges, panel, cv::COLOR_GRAY2BGR);
    const int w = edges.cols, h = edges.rows;
    const int top = cvRound(h * 0.58), bottom = h - 1;
    std::vector<cv::Point> roi = {{cvRound(w * .08), bottom},
        {cvRound(w * .43), top}, {cvRound(w * .57), top},
        {cvRound(w * .92), bottom}};
    cv::Mat mask = cv::Mat::zeros(edges.size(), CV_8UC1);
    cv::fillConvexPoly(mask, roi, cv::Scalar(255));
    cv::threshold(edges, edges, 80, 255, cv::THRESH_BINARY);
    cv::bitwise_and(edges, mask, edges);
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 20,
                    std::max(12.0, h * .08), std::max(8.0, h * .04));
    // Length-weighted x(y) fits avoid unstable division by small dx.
    double weights[2] = {}, slopes[2] = {}, offsets[2] = {};
    for (const auto& l : lines) {
        const double dx = l[2] - l[0], dy = l[3] - l[1];
        if (std::abs(dx) < 1 || std::abs(dy) < 1) continue;
        const double slope = dy / dx;
        if (std::abs(slope) < .45 || std::abs(slope) > 3.5) continue;
        const double a = dx / dy, b = l[0] - a * l[1];
        const double xb = a * bottom + b, xt = a * top + b;
        const int side = slope < 0 ? 0 : 1;
        const double midpoint = (l[0] + l[2]) * .5;
        if (xt < .30 * w || xt > .70 * w) continue;
        if (side == 0 && (midpoint > .53*w || xb < .05*w || xb > .48*w)) continue;
        if (side == 1 && (midpoint < .47*w || xb < .52*w || xb > .95*w)) continue;
        const double weight = std::hypot(dx, dy);
        weights[side] += weight;
        slopes[side] += a * weight;
        offsets[side] += b * weight;
    }
    cv::polylines(panel, std::vector<std::vector<cv::Point>>{roi}, true,
                  cv::Scalar(110, 100, 75), 1, cv::LINE_AA);
    for (int side = 0; side < 2; ++side) {
        if (weights[side] == 0) continue; // Do not invent a missing lane.
        const double a = slopes[side] / weights[side];
        const double b = offsets[side] / weights[side];
        const cv::Point p1(cvRound(a * bottom + b), bottom);
        const cv::Point p2(cvRound(a * top + b), top);
        cv::line(panel, p1, p2, cv::Scalar(10, 10, 10), 9, cv::LINE_AA);
        cv::line(panel, p1, p2, side == 0 ? cv::Scalar(255, 210, 40) :
                 cv::Scalar(60, 220, 255), 5, cv::LINE_AA);
    }
    caption(panel, weights[0] ? "LEFT candidate" : "LEFT: none", 8, 20, .45,
            cv::Scalar(255, 210, 40));
    caption(panel, weights[1] ? "RIGHT candidate" : "RIGHT: none", w / 2, 20, .45,
            cv::Scalar(60, 220, 255));
}

class LocalPreview {
public:
    // Probe before VART/OpenCV worker initialization. Some GUI backends abort
    // instead of throwing when DISPLAY is stale; isolate that startup failure.
    void initialize(bool requested) {
        if (!requested) return;
        const char* display = std::getenv("DISPLAY");
        const char* wayland = std::getenv("WAYLAND_DISPLAY");
        if ((!display || !*display) && (!wayland || !*wayland)) {
            unavailable(); return;
        }
        const pid_t child = ::fork();
        if (child == 0) {
            struct rlimit limit = {0, 0};
            ::setrlimit(RLIMIT_CORE, &limit);
            std::signal(SIGALRM, SIG_DFL);
            ::alarm(5);
            try {
                cv::namedWindow(name, cv::WINDOW_NORMAL);
                cv::imshow(name, cv::Mat::zeros(32, 32, CV_8UC3));
                cv::waitKey(1);
                cv::destroyAllWindows();
                ::_exit(0);
            } catch (...) { ::_exit(1); }
        }
        int status = 0;
        pid_t result = -1;
        if (child > 0) {
            do { result = ::waitpid(child, &status, 0); }
            while (result < 0 && errno == EINTR);
        }
        if (result != child || child < 0 || !WIFEXITED(status) || WEXITSTATUS(status)) {
            unavailable(); return;
        }
        try {
            cv::namedWindow(name, cv::WINDOW_NORMAL);
            enabled = true;
            std::cout << "Local camera preview enabled; q/Q stops cleanly\n";
        } catch (const cv::Exception&) { unavailable(); }
    }
    void show(const cv::Mat& frame) {
        if (!enabled) return;
        try {
            cv::imshow(name, frame);
            const int key = cv::waitKey(1) & 0xff;
            if (key == 'q' || key == 'Q') g_stop_requested = 1;
        } catch (const cv::Exception&) { close(); unavailable(); }
    }
    ~LocalPreview() { close(); }
private:
    void close() {
        if (enabled) {
            try { cv::destroyWindow(name); } catch (const cv::Exception&) {}
            enabled = false;
        }
    }
    void unavailable() {
        std::cout << "Local GUI unavailable/headless; continuing processing and optional recording.\n";
    }
    const char* name = "KV260 Autonomous Vehicle Vision";
    bool enabled = false;
};

int main(int argc, char* argv[])
{
    std::signal(
        SIGINT,
        handle_stop_signal);

    std::signal(
        SIGTERM,
        handle_stop_signal);

    std::cout
        << "========================================\n"
        << " KV260 HLS + DPU Video Perception\n"
        << "========================================\n";

    /*
     * Supported modes:
     *
     * Video file:
     *   app MODEL input.mp4 output.avi
     *
     * Video file + live browser stream:
     *   app MODEL input.mp4 output.avi --stream 8080
     *
     * USB camera:
     *   app MODEL --camera 0 output.avi
     *
     * USB camera + live browser stream:
     *   app MODEL --camera 0 output.avi --stream 8080
     */
    const bool camera_mode =
        ((argc == 5 ||
          argc == 7) &&
         std::string(argv[2]) == "--camera");

    const bool video_mode =
        ((argc == 4 ||
          argc == 6) &&
         std::string(argv[2]) != "--camera");

    if (!camera_mode &&
        !video_mode) {

        std::cerr
            << "Usage:\n"
            << "  Video file:\n"
            << "    "
            << argv[0]
            << " <model.xmodel> <input_video> <output.avi>\n"
            << "\n"
            << "  Video file + live stream:\n"
            << "    "
            << argv[0]
            << " <model.xmodel> <input_video> <output.avi>"
            << " --stream <port>\n"
            << "\n"
            << "  USB camera:\n"
            << "    "
            << argv[0]
            << " <model.xmodel> --camera <index> <output.avi>\n"
            << "\n"
            << "  USB camera + live stream:\n"
            << "    "
            << argv[0]
            << " <model.xmodel> --camera <index> <output.avi>"
            << " --stream <port>\n"
            << "Camera output may be - to disable AVI recording.\n";

        return 1;
    }

    const std::string model_path =
        argv[1];

    std::string video_path;

    int camera_index = -1;

    std::string output_path;

    bool stream_enabled = false;
    int stream_port = 0;

    if (camera_mode) {

        try {
            camera_index =
                std::stoi(
                    argv[3]);

        } catch (...) {

            std::cerr
                << "ERROR: Invalid camera index: "
                << argv[3]
                << "\n";

            return 1;
        }

        if (camera_index < 0) {
            std::cerr
                << "ERROR: Camera index must be >= 0\n";
            return 1;
        }

        output_path =
            argv[4];

        if (argc == 7) {

            if (std::string(argv[5]) !=
                "--stream") {

                std::cerr
                    << "ERROR: Expected --stream before port\n";

                return 1;
            }

            try {
                stream_port =
                    std::stoi(
                        argv[6]);

            } catch (...) {

                std::cerr
                    << "ERROR: Invalid stream port: "
                    << argv[6]
                    << "\n";

                return 1;
            }

            stream_enabled = true;
        }

    } else {

        video_path =
            argv[2];

        output_path =
            argv[3];

        if (argc == 6) {

            if (std::string(argv[4]) !=
                "--stream") {

                std::cerr
                    << "ERROR: Expected --stream before port\n";

                return 1;
            }

            try {
                stream_port =
                    std::stoi(
                        argv[5]);

            } catch (...) {

                std::cerr
                    << "ERROR: Invalid stream port: "
                    << argv[5]
                    << "\n";

                return 1;
            }

            stream_enabled = true;
        }
    }

    if (stream_enabled &&
        (stream_port <= 0 ||
         stream_port > 65535)) {

        std::cerr
            << "ERROR: Stream port must be 1..65535\n";

        return 1;
    }

    LocalPreview preview;
    preview.initialize(camera_mode);

    /*
     * ---------------------------------------------------------
     * DPU INITIALIZATION
     * Done ONCE for the whole video.
     * ---------------------------------------------------------
     */
    auto graph =
        xir::Graph::deserialize(
            model_path);

    if (!graph) {
        std::cerr
            << "ERROR: Cannot deserialize XModel\n";
        return 1;
    }

    auto root =
        graph->get_root_subgraph();

    const xir::Subgraph*
        dpu_subgraph = nullptr;

    for (auto* sg :
         root->children_topological_sort()) {

        if (!sg->has_attr("device")) {
            continue;
        }

        if (sg->get_attr<std::string>(
                "device") == "DPU") {

            dpu_subgraph = sg;
            break;
        }
    }

    if (dpu_subgraph == nullptr) {
        std::cerr
            << "ERROR: DPU subgraph not found\n";
        return 1;
    }

    auto attrs =
        xir::Attrs::create();

    auto runner =
        vart::RunnerExt::create_runner(
            dpu_subgraph,
            attrs.get());

    if (!runner) {
        std::cerr
            << "ERROR: RunnerExt creation failed\n";
        return 1;
    }

    auto inputs =
        runner->get_inputs();

    auto outputs =
        runner->get_outputs();

    if (inputs.size() != 1 ||
        outputs.size() != 3) {

        std::cerr
            << "ERROR: Expected 1 input and 3 outputs\n";
        return 1;
    }

    auto* input_tensor =
        inputs[0]->get_tensor();

    const auto input_shape =
        input_tensor->get_shape();

    if (input_shape.size() != 4) {
        std::cerr
            << "ERROR: Input tensor is not NHWC\n";
        return 1;
    }

    const int batch =
        input_shape[0];

    const int network_height =
        input_shape[1];

    const int network_width =
        input_shape[2];

    const int channels =
        input_shape[3];

    if (batch != 1 ||
        channels != 3) {

        std::cerr
            << "ERROR: Expected [1,H,W,3] input\n";
        return 1;
    }

    if (!input_tensor->has_attr(
            "fix_point")) {

        std::cerr
            << "ERROR: Input tensor has no fix_point\n";
        return 1;
    }

    const int input_fix =
        input_tensor->get_attr<int>(
            "fix_point");

    const float input_fixed_scale =
        std::ldexp(
            1.0f,
            input_fix);

    const float real_scale =
        MODEL_SCALE *
        input_fixed_scale;

    uint64_t input_address = 0;
    std::size_t input_size = 0;

    std::tie(
        input_address,
        input_size) =
        inputs[0]->data(
            zero_index(
                input_tensor));

    if (input_address == 0 ||
        input_size == 0) {

        std::cerr
            << "ERROR: Cannot access DPU input buffer\n";
        return 1;
    }

    auto* input_data =
        reinterpret_cast<int8_t*>(
            input_address);

    const std::size_t
        input_bytes_per_batch =
            input_tensor->get_data_size() /
            static_cast<std::size_t>(
                batch);

    std::cout
        << "DPU initialized once\n"
        << "Network input = "
        << network_width
        << "x"
        << network_height
        << "\n"
        << "Input fix_point = "
        << input_fix
        << "\n";

    /*
     * ---------------------------------------------------------
     * VIDEO INITIALIZATION
     * ---------------------------------------------------------
     */
    cv::VideoCapture capture;

    if (camera_mode) {

        /*
         * Prefer the Linux V4L2 backend for a USB camera.
         * Fall back to OpenCV automatic backend selection
         * if V4L2 cannot open the device.
         */
        capture.open(
            camera_index,
            cv::CAP_V4L2);

        if (!capture.isOpened()) {
            capture.open(
                camera_index);
        }

        if (!capture.isOpened()) {
            std::cerr
                << "ERROR: Cannot open camera index "
                << camera_index
                << "\n";

            return 1;
        }

        std::cout
            << "Camera input = /dev/video"
            << camera_index
            << "\n";

    } else {

        capture.open(
            video_path);

        if (!capture.isOpened()) {
            std::cerr
                << "ERROR: Cannot open video: "
                << video_path
                << "\n";

            return 1;
        }

        std::cout
            << "Video input = "
            << video_path
            << "\n";
    }

    double source_fps =
        capture.get(
            cv::CAP_PROP_FPS);

    if (!(source_fps > 1.0 &&
          source_fps < 240.0)) {

        source_fps = 30.0;
    }

    cv::Mat frame;
    const auto first_acquisition_start = std::chrono::steady_clock::now();

    if (!capture.read(frame) ||
        frame.empty()) {

        std::cerr
            << "ERROR: Video contains no readable frames\n";
        return 1;
    }

    if (frame.channels() != 3) {
        std::cerr
            << "ERROR: Expected 3-channel BGR video frames\n";
        return 1;
    }

    const double first_acquisition_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - first_acquisition_start).count();
    const cv::Size source_frame_size = frame.size();
    const bool recording = !(camera_mode && output_path == "-");
    cv::VideoWriter writer;

    if (recording) writer.open(
        output_path,
        cv::VideoWriter::fourcc(
            'M', 'J', 'P', 'G'),
        source_fps,
        PRESENTATION_SIZE,
        true);

    if (recording && !writer.isOpened()) {
        std::cerr
            << "ERROR: Cannot create output video: "
            << output_path
            << "\n"
            << "Use an .avi output filename for MJPG.\n";
        return 1;
    }

    std::cout
        << "Input mode  = "
        << (camera_mode
                ? "USB CAMERA"
                : "VIDEO FILE")
        << "\n"
        << "Frame size  = "
        << frame.cols
        << "x"
        << frame.rows
        << "\n"
        << "Source FPS  = "
        << source_fps
        << "\n"
        << "Output      = "
        << output_path
        << "\n";

    MjpegHttpServer stream_server;

    if (stream_enabled) {

        if (!stream_server.start(
                stream_port)) {

            std::cerr
                << "ERROR: Cannot start MJPEG live stream\n";

            return 1;
        }
    }

    /*
     * ---------------------------------------------------------
     * SOBEL INITIALIZATION
     * Open /dev/sobel_accel and mmap ONCE.
     *
     * If Sobel is unavailable, YOLO video inference continues.
     * ---------------------------------------------------------
     */
    SobelHardwareContext sobel_hardware;

    bool sobel_enabled =
        sobel_hardware.initialize();

    if (!sobel_enabled) {
        std::cerr
            << "WARNING: HLS Sobel unavailable. "
            << "Continuing with DPU video inference.\n";
    }

    std::size_t processed_frames = 0;

    double total_hls_sobel_ms = 0.0;
    double total_preprocess_ms = 0.0;
    double total_dpu_ms = 0.0;
    double total_postprocess_ms = 0.0;
    double total_compute_ms = 0.0;
    double total_end_to_end_ms = 0.0;
    double previous_end_to_end_fps = 0.0;
    cv::Mat result_frame(PRESENTATION_SIZE, CV_8UC3);

    /*
     * =========================================================
     * CONTINUOUS VIDEO LOOP
     * =========================================================
     */
    while (!g_stop_requested) {
        const auto frame_start = std::chrono::steady_clock::now();
        if (processed_frames != 0 && !capture.read(frame)) break;
        if (frame.empty()) {
            break;
        }

        if (frame.channels() != 3) {
            std::cerr
                << "WARNING: Stopping at non-BGR frame "
                << processed_frames
                << "\n";
            break;
        }

        /*
         * ---------------- HLS SOBEL BRANCH ----------------
         */
        double hls_sobel_ms = 0.0;
        cv::Mat sobel_frame;

        if (sobel_enabled) {

            if (!sobel_hardware.process(
                    frame,
                    sobel_frame,
                    hls_sobel_ms)) {

                std::cerr
                    << "WARNING: HLS Sobel failed on frame "
                    << processed_frames
                    << ". Disabling Sobel branch.\n";

                sobel_enabled = false;
                hls_sobel_ms = 0.0;
            }
        }

        /*
         * ---------------- DPU PREPROCESSING ----------------
         *
         * Preserve the proven static application preprocessing:
         * direct resize + BGR -> RGB + INT8 quantization.
         */
        const auto preprocess_start =
            std::chrono::steady_clock::now();

        cv::Mat resized;

        cv::resize(
            frame,
            resized,
            cv::Size(
                network_width,
                network_height),
            0.0,
            0.0,
            cv::INTER_LINEAR);

        for (int h = 0;
             h < network_height;
             ++h) {

            const uint8_t* row =
                resized.ptr<uint8_t>(h);

            for (int w = 0;
                 w < network_width;
                 ++w) {

                const uint8_t b =
                    row[w * 3 + 0];

                const uint8_t g =
                    row[w * 3 + 1];

                const uint8_t r =
                    row[w * 3 + 2];

                const std::size_t index =
                    (static_cast<std::size_t>(h) *
                         network_width +
                     w) *
                    3;

                input_data[index + 0] =
                    static_cast<int8_t>(
                        static_cast<int>(
                            r * real_scale));

                input_data[index + 1] =
                    static_cast<int8_t>(
                        static_cast<int>(
                            g * real_scale));

                input_data[index + 2] =
                    static_cast<int8_t>(
                        static_cast<int>(
                            b * real_scale));
            }
        }

        inputs[0]->sync_for_write(
            0,
            input_bytes_per_batch);

        const auto preprocess_end =
            std::chrono::steady_clock::now();

        const double preprocess_ms =
            std::chrono::duration<
                double,
                std::milli>(
                    preprocess_end -
                    preprocess_start)
                .count();

        /*
         * ---------------- DPU INFERENCE ----------------
         */
        const auto dpu_start =
            std::chrono::steady_clock::now();

        auto job =
            runner->execute_async(
                inputs,
                outputs);

        const int status =
            runner->wait(
                static_cast<int>(
                    job.first),
                -1);

        const auto dpu_end =
            std::chrono::steady_clock::now();

        const double dpu_ms =
            std::chrono::duration<
                double,
                std::milli>(
                    dpu_end -
                    dpu_start)
                .count();

        if (status != 0) {
            std::cerr
                << "ERROR: DPU failed on frame "
                << processed_frames
                << " status="
                << status
                << "\n";

            return 1;
        }

        /*
         * ---------------- YOLO POSTPROCESSING ----------------
         */
        const auto postprocess_start =
            std::chrono::steady_clock::now();

        for (auto* output : outputs) {

            auto* tensor =
                output->get_tensor();

            const auto shape =
                tensor->get_shape();

            const std::size_t
                bytes_per_batch =
                    tensor->get_data_size() /
                    static_cast<std::size_t>(
                        shape[0]);

            output->sync_for_read(
                0,
                bytes_per_batch);
        }

        std::vector<YoloBox> boxes;

        for (auto* output : outputs) {

            if (!decode_output_tensor(
                    output,
                    network_width,
                    network_height,
                    boxes)) {

                std::cerr
                    << "ERROR: YOLO decode failed on frame "
                    << processed_frames
                    << "\n";

                return 1;
            }
        }

        auto detections =
            apply_vitis_ai_nms(
                boxes);

        const auto postprocess_end =
            std::chrono::steady_clock::now();

        const double postprocess_ms =
            std::chrono::duration<
                double,
                std::milli>(
                    postprocess_end -
                    postprocess_start)
                .count();

        /*
         * ---------------- DRAW YOLO RESULTS ----------------
         */
        result_frame.setTo(cv::Scalar(24, 20, 16));
        caption(result_frame, "KV260 AUTONOMOUS VEHICLE VISION", 20, 31, .85);
        caption(result_frame, "HLS SOBEL + VITIS AI DPU", 20, 57, .55, {210, 190, 130});
        caption(result_frame, "YOLO OBJECT DETECTION", 16, 88, .65);
        caption(result_frame, "HLS SOBEL / LANE VIEW", 648, 88, .65);
        const cv::Rect left = fitted_rect(frame.size(), 12);
        const cv::Rect right = fitted_rect(frame.size(), 644);
        cv::Mat color_panel = result_frame(left);
        cv::resize(frame, color_panel, left.size());
        // Original frame -> YOLO: only object annotations on the color panel.
        for (const auto& d : detections) {
            const int x1 = std::max(0, std::min(left.width - 1, int(d.x * left.width)));
            const int y1 = std::max(0, std::min(left.height - 1, int(d.y * left.height)));
            const int x2 = std::max(0, std::min(left.width - 1, int((d.x+d.width)*left.width)));
            const int y2 = std::max(0, std::min(left.height - 1, int((d.y+d.height)*left.height)));
            if (x2 <= x1 || y2 <= y1) continue;
            cv::rectangle(color_panel, {x1, y1}, {x2, y2}, {70, 230, 110}, 2);
            std::ostringstream label;
            label << COCO_CLASSES[d.class_id] << " " << std::fixed
                  << std::setprecision(0) << d.score * 100 << "%";
            int baseline = 0;
            const cv::Size text = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX,
                                                 .48, 1, &baseline);
            const int tx = std::max(0, std::min(x1, left.width - text.width - 6));
            const int ty = std::max(text.height + 6, y1);
            cv::rectangle(color_panel, {tx, ty - text.height - 6},
                          {tx + text.width + 6, ty + baseline}, {20, 45, 25}, cv::FILLED);
            caption(color_panel, label.str(), tx + 3, ty - 3, .48, {100, 255, 150});
        }
        // HLS Sobel -> lane candidates, never a software substitute for HLS edges.
        if (sobel_enabled && !sobel_frame.empty()) lane_view(sobel_frame, result_frame(right));
        else caption(result_frame, "HLS Sobel unavailable", right.x + 20, right.y + 40);
        caption(result_frame, "HLS edges + ARM/OpenCV lane candidates", 648, 471, .48);

        const double compute_ms =
            hls_sobel_ms +
            preprocess_ms +
            dpu_ms +
            postprocess_ms;

        const double compute_fps =
            (compute_ms > 0.0)
                ? (1000.0 / compute_ms)
                : 0.0;

        std::ostringstream status_bar;
        status_bar << std::fixed << std::setprecision(2)
                   << "HLS: " << hls_sobel_ms << " ms | DPU: " << dpu_ms
                   << " ms | Pre: " << preprocess_ms << " ms | Post: " << postprocess_ms
                   << " ms | Compute FPS: " << std::setprecision(1) << compute_fps
                   << " | Objects: " << detections.size();
        caption(result_frame, status_bar.str(), 16, 502, .53);
        caption(result_frame, processed_frames == 0 ? "End-to-end FPS (previous): warming up" :
                "End-to-end FPS (previous): " + cv::format("%.1f", previous_end_to_end_fps),
                16, 471, .48);
        if (recording) writer.write(result_frame);
        // Includes acquisition, full Sobel branch, inference, lanes, drawing and AVI write.
        // First read precedes one-time setup; count its duration without setup overhead.
        const double end_to_end_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - frame_start).count() +
            (processed_frames == 0 ? first_acquisition_ms : 0.0);
        previous_end_to_end_fps = end_to_end_ms > 0 ? 1000.0 / end_to_end_ms : 0.0;
        total_end_to_end_ms += end_to_end_ms;
        preview.show(result_frame);

        if (stream_enabled) {
            stream_server.publish(
                result_frame);
        }

        total_hls_sobel_ms +=
            hls_sobel_ms;

        total_preprocess_ms +=
            preprocess_ms;

        total_dpu_ms +=
            dpu_ms;

        total_postprocess_ms +=
            postprocess_ms;

        total_compute_ms +=
            compute_ms;

        ++processed_frames;

        if (processed_frames == 1 ||
            processed_frames % 30 == 0) {

            std::cout
                << "Frame "
                << processed_frames
                << " | detections="
                << detections.size()
                << " | HLS="
                << hls_sobel_ms
                << " ms"
                << " | DPU="
                << dpu_ms
                << " ms"
                << " | compute FPS="
                << compute_fps
                << " | end-to-end FPS=" << previous_end_to_end_fps
                << "\n";
        }

    }

    if (g_stop_requested) {
        std::cout
            << "\nStop requested - shutting down cleanly\n";
    }

    capture.release();
    writer.release();

    if (processed_frames == 0) {
        std::cerr
            << "ERROR: No video frames processed\n";
        return 1;
    }

    const double frames =
        static_cast<double>(
            processed_frames);

    const double avg_hls_sobel_ms =
        total_hls_sobel_ms /
        frames;

    const double avg_preprocess_ms =
        total_preprocess_ms /
        frames;

    const double avg_dpu_ms =
        total_dpu_ms /
        frames;

    const double avg_postprocess_ms =
        total_postprocess_ms /
        frames;

    const double avg_compute_ms =
        total_compute_ms /
        frames;

    const double avg_compute_fps =
        (avg_compute_ms > 0.0)
            ? (1000.0 /
               avg_compute_ms)
            : 0.0;

    std::cout
        << "\n========================================\n"
        << " VIDEO BENCHMARK SUMMARY\n"
        << "========================================\n"
        << "Frames processed       = "
        << processed_frames
        << "\n"
        << "Average HLS Sobel       = "
        << avg_hls_sobel_ms
        << " ms\n"
        << "Average preprocessing   = "
        << avg_preprocess_ms
        << " ms\n"
        << "Average DPU execution   = "
        << avg_dpu_ms
        << " ms\n"
        << "Average postprocessing  = "
        << avg_postprocess_ms
        << " ms\n"
        << "Average compute/frame   = "
        << avg_compute_ms
        << " ms\n"
        << "Average compute FPS     = "
        << avg_compute_fps
        << "\n"
        << "Average end-to-end/frame = " << total_end_to_end_ms / frames << " ms\n"
        << "Average end-to-end FPS   = " << 1000.0 * frames / total_end_to_end_ms << "\n"
        << "End-to-end excludes preview/HTTP delivery and one-time setup.\n"
        << "Output video            = "
        << output_path
        << "\n"
        << "========================================\n";

    /*
     * Save one summary row per video run.
     */
    const std::string benchmark_csv =
        "video_presentation_benchmark_results.csv";

    bool write_header = false;

    {
        std::ifstream existing(
            benchmark_csv,
            std::ios::binary |
            std::ios::ate);

        write_header =
            !existing.good() ||
            existing.tellg() == 0;
    }

    std::ofstream csv(
        benchmark_csv,
        std::ios::app);

    if (csv) {

        if (write_header) {
            csv
                << "video,"
                << "frames,"
                << "width,"
                << "height,"
                << "source_fps,"
                << "avg_hls_sobel_ms,"
                << "avg_preprocess_ms,"
                << "avg_dpu_ms,"
                << "avg_postprocess_ms,"
                << "avg_compute_ms,"
                << "avg_compute_fps,avg_end_to_end_ms,avg_end_to_end_fps\n";
        }

        csv
            << std::fixed
            << std::setprecision(6)
            << (camera_mode
                    ? ("camera:" +
                       std::to_string(
                           camera_index))
                    : video_path)
            << ","
            << processed_frames << ","
            << source_frame_size.width << ","
            << source_frame_size.height << ","
            << source_fps << ","
            << avg_hls_sobel_ms << ","
            << avg_preprocess_ms << ","
            << avg_dpu_ms << ","
            << avg_postprocess_ms << ","
            << avg_compute_ms << ","
            << avg_compute_fps << ","
            << total_end_to_end_ms / frames << ","
            << 1000.0 * frames / total_end_to_end_ms
            << "\n";
    }

    return 0;
}

