#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <chrono>
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

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <xir/attrs/attrs.hpp>
#include <xir/graph/graph.hpp>
#include <xir/graph/subgraph.hpp>
#include <xir/tensor/tensor.hpp>

#include <vart/runner_ext.hpp>
#include <vart/tensor_buffer.hpp>

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

    /*
     * TEMPORARY YOLO RAW OUTPUT DIAGNOSTIC.
     *
     * This does not modify inference or decoding.
     * It only measures the actual INT8 tensor values
     * produced by the DPU.
     */
    {
        const std::size_t total_values =
            static_cast<std::size_t>(height) *
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(channels);

        int raw_min = 127;
        int raw_max = -128;

        for (std::size_t i = 0;
             i < total_values;
             ++i) {

            const int value =
                static_cast<int>(result[i]);

            if (value < raw_min) {
                raw_min = value;
            }

            if (value > raw_max) {
                raw_max = value;
            }
        }

        int object_raw_min = 127;
        int object_raw_max = -128;

        float object_logit_min = 1.0e30f;
        float object_logit_max = -1.0e30f;

        std::size_t object_count = 0;
        std::size_t object_pass_count = 0;

        for (int dh = 0; dh < height; ++dh) {

            for (int dw = 0; dw < width; ++dw) {

                for (int da = 0;
                     da < ANCHORS_PER_SCALE;
                     ++da) {

                    const int diagnostic_index =
                        ((dh * width + dw) *
                         ANCHORS_PER_SCALE +
                         da) *
                        values_per_anchor;

                    const int raw_objectness =
                        static_cast<int>(
                            result[
                                diagnostic_index +
                                4]);

                    const float object_logit =
                        static_cast<float>(
                            raw_objectness) *
                        dequant_scale;

                    if (raw_objectness <
                        object_raw_min) {
                        object_raw_min =
                            raw_objectness;
                    }

                    if (raw_objectness >
                        object_raw_max) {
                        object_raw_max =
                            raw_objectness;
                    }

                    if (object_logit <
                        object_logit_min) {
                        object_logit_min =
                            object_logit;
                    }

                    if (object_logit >
                        object_logit_max) {
                        object_logit_max =
                            object_logit;
                    }

                    ++object_count;

                    if (object_logit >=
                        conf_desigmoid) {

                        ++object_pass_count;
                    }
                }
            }
        }

        std::cout
            << "\n========================================\n"
            << " YOLO RAW OUTPUT DIAGNOSTIC\n"
            << "========================================\n"
            << "TENSOR = "
            << tensor->get_name()
            << "\n"
            << "SHAPE = ";

        print_shape(tensor);

        std::cout
            << "\nFIX_POINT = "
            << fix_point
            << "\nDEQUANT_SCALE = "
            << dequant_scale
            << "\nRAW INT8 MIN = "
            << raw_min
            << "\nRAW INT8 MAX = "
            << raw_max
            << "\nOBJECTNESS RAW MIN = "
            << object_raw_min
            << "\nOBJECTNESS RAW MAX = "
            << object_raw_max
            << "\nOBJECTNESS LOGIT MIN = "
            << object_logit_min
            << "\nOBJECTNESS LOGIT MAX = "
            << object_logit_max
            << "\nOBJECTNESS >= THRESHOLD LOGIT = "
            << object_pass_count
            << " / "
            << object_count
            << "\n========================================\n";
    }

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

static bool run_sobel_hardware(
    const cv::Mat& image,
    const std::string& output_path,
    double& sobel_hw_ms)
{
    if (image.empty()) {
        std::cerr
            << "SOBEL ERROR: empty input image\n";
        return false;
    }

    if (image.cols <= 0 ||
        image.rows <= 0 ||
        static_cast<std::size_t>(image.cols) >
            SOBEL_MAX_WIDTH ||
        static_cast<std::size_t>(image.rows) >
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

    const std::size_t buffer_stride =
        ((SOBEL_MAX_FRAME_BYTES +
          page_size - 1) /
         page_size) *
        page_size;

    const std::size_t total_dma_bytes =
        2U * buffer_stride;

    const std::size_t frame_bytes =
        static_cast<std::size_t>(
            image.cols) *
        static_cast<std::size_t>(
            image.rows);

    int fd =
        ::open(
            "/dev/sobel_accel",
            O_RDWR);

    if (fd < 0) {
        std::cerr
            << "SOBEL ERROR: open(/dev/sobel_accel): "
            << std::strerror(errno)
            << "\n";

        return false;
    }

    void* mapped =
        ::mmap(
            nullptr,
            total_dma_bytes,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd,
            0);

    if (mapped == MAP_FAILED) {
        std::cerr
            << "SOBEL ERROR: mmap(): "
            << std::strerror(errno)
            << "\n";

        ::close(fd);
        return false;
    }

    auto* input_dma =
        static_cast<std::uint8_t*>(
            mapped);

    auto* output_dma =
        input_dma + buffer_stride;

    cv::Mat gray;

    cv::cvtColor(
        image,
        gray,
        cv::COLOR_BGR2GRAY);

    /*
     * Copy grayscale pixels into the coherent
     * Sobel input DMA buffer.
     */
    for (int row = 0;
         row < gray.rows;
         ++row) {

        std::memcpy(
            input_dma +
                static_cast<std::size_t>(row) *
                static_cast<std::size_t>(gray.cols),

            gray.ptr<std::uint8_t>(row),

            static_cast<std::size_t>(
                gray.cols));
    }

    /*
     * Clear only the active output frame area.
     */
    std::memset(
        output_dma,
        0,
        frame_bytes);

    SobelRunRequest request{};

    request.width =
        static_cast<std::uint32_t>(
            image.cols);

    request.height =
        static_cast<std::uint32_t>(
            image.rows);

    std::cout
        << "\nSOBEL HLS\n"
        << "Input  = "
        << request.width
        << "x"
        << request.height
        << "\n"
        << "DMA map = "
        << total_dma_bytes
        << " bytes\n";

    const auto sobel_hw_start =
        std::chrono::steady_clock::now();

    const int ioctl_result =
        ::ioctl(
            fd,
            SOBEL_IOCTL_RUN,
            &request);

    const auto sobel_hw_end =
        std::chrono::steady_clock::now();

    sobel_hw_ms =
        std::chrono::duration<double, std::milli>(
            sobel_hw_end -
            sobel_hw_start).count();

    std::cout
        << "Sobel HLS execution = "
        << sobel_hw_ms
        << " ms\\n";

    if (ioctl_result < 0) {
        std::cerr
            << "SOBEL ERROR: ioctl(SOBEL_RUN): "
            << std::strerror(errno)
            << "\n";

        ::munmap(
            mapped,
            total_dma_bytes);

        ::close(fd);

        return false;
    }

    /*
     * Create an OpenCV view of the hardware
     * output buffer.
     *
     * The HLS accelerator writes one byte/pixel.
     */
    cv::Mat sobel_output(
        image.rows,
        image.cols,
        CV_8UC1,
        output_dma,
        static_cast<std::size_t>(
            image.cols));

    const bool saved =
        cv::imwrite(
            output_path,
            sobel_output);

    if (!saved) {
        std::cerr
            << "SOBEL ERROR: cannot save "
            << output_path
            << "\n";
    } else {
        std::cout
            << "Sobel output saved = "
            << output_path
            << "\n";
    }

    ::munmap(
        mapped,
        total_dma_bytes);

    ::close(fd);

    return saved;
}

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

int main(int argc, char* argv[])
{
    std::cout
        << "========================================\n"
        << " KV260 YOLOv5 Nano Detection Test\n"
        << "========================================\n";

    if (argc != 3) {
        std::cerr
            << "Usage:\n"
            << "  "
            << argv[0]
            << " <model.xmodel> <image.jpg>\n";

        return 1;
    }

    const std::string model_path =
        argv[1];

    const std::string image_path =
        argv[2];

    cv::Mat image =
        cv::imread(
            image_path,
            cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr
            << "ERROR: Cannot load image: "
            << image_path
            << "\n";

        return 1;
    }

    std::cout
        << "Original image = "
        << image.cols
        << "x"
        << image.rows
        << "\n";


    double cpu_sobel_ms = 0.0;

    if (!run_sobel_cpu_reference(
            image,
            "cpu_sobel_result.jpg",
            cpu_sobel_ms)) {

        std::cerr
            << "WARNING: CPU Sobel reference failed\n";
    }

    /*
     * Run the HLS Sobel branch on the original image.
     *
     * A Sobel failure does not stop the YOLO branch.
     */
    double hls_sobel_ms = 0.0;

    if (!run_sobel_hardware(
            image,
            "sobel_result.jpg",
            hls_sobel_ms)) {

        std::cerr
            << "WARNING: Sobel branch failed; "
            << "continuing with YOLO DPU\n";
    }

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

    std::cout << "\nINPUT\n";
    std::cout
        << "NAME  = "
        << input_tensor->get_name()
        << "\n";

    std::cout
        << "SHAPE = ";

    print_shape(input_tensor);

    std::cout << "\n";

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

    std::cout
        << "FIX_POINT = "
        << input_fix
        << "\n";

    std::cout
        << "INPUT QUANTIZATION SCALE = "
        << real_scale
        << "\n";

    const auto preprocess_start =
        std::chrono::steady_clock::now();

    /*
     * Match normal Vitis AI YOLOv5 path:
     * direct resize to network size.
     */
    cv::Mat resized;

    cv::resize(
        image,
        resized,
        cv::Size(
            network_width,
            network_height),
        0.0,
        0.0,
        cv::INTER_LINEAR);

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
            << "ERROR: Cannot access input buffer\n";

        return 1;
    }

    auto* input_data =
        reinterpret_cast<int8_t*>(
            input_address);

    /*
     * Reproduce NormalizeInputDataRGB().
     *
     * OpenCV BGR -> DPU RGB
     * integer truncation path.
     */
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

    std::cout
        << "IMAGE PREPROCESSING = OK\n";

    const std::size_t
        input_bytes_per_batch =
            input_tensor->get_data_size() /
            static_cast<std::size_t>(
                batch);

    inputs[0]->sync_for_write(
        0,
        input_bytes_per_batch);

    const auto preprocess_end =
        std::chrono::steady_clock::now();

    const double preprocess_ms =
        std::chrono::duration<double, std::milli>(
            preprocess_end -
            preprocess_start).count();

    std::cout
        << "YOLO preprocessing = "
        << preprocess_ms
        << " ms\\n";

    std::cout
        << "\nStarting DPU...\n";

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
        std::chrono::duration<double, std::milli>(
            dpu_end -
            dpu_start).count();

    if (status != 0) {
        std::cerr
            << "ERROR: DPU execution failed"
            << " status="
            << status
            << "\n";

        return 1;
    }

    std::cout
        << "DPU EXECUTION = OK\n"
        << "DPU execution = "
        << dpu_ms
        << " ms\\n";

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

    /*
     * We intentionally identify each
     * anchor group by tensor dimensions,
     * so RunnerExt output ordering
     * does not matter.
     */
    for (auto* output : outputs) {

        if (!decode_output_tensor(
                output,
                network_width,
                network_height,
                boxes)) {

            return 1;
        }
    }

    std::cout
        << "\nCandidate boxes after objectness filter = "
        << boxes.size()
        << "\n";

    auto detections =
        apply_vitis_ai_nms(
            boxes);

    const auto postprocess_end =
        std::chrono::steady_clock::now();

    const double postprocess_ms =
        std::chrono::duration<double, std::milli>(
            postprocess_end -
            postprocess_start).count();

    const double sequential_compute_ms =
        hls_sobel_ms +
        preprocess_ms +
        dpu_ms +
        postprocess_ms;

    std::cout
        << "Final detections after NMS = "
        << detections.size()
        << "\n";

    std::cout
        << "\n========================================\n"
        << " DETECTIONS\n"
        << "========================================\n";

    for (std::size_t i = 0;
         i < detections.size();
         ++i) {

        const auto& d =
            detections[i];

        std::cout
            << "Detection "
            << i
            << "\n";

        std::cout
            << "  CLASS_ID = "
            << d.class_id
            << "\n";

        std::cout
            << "  CLASS    = "
            << COCO_CLASSES[d.class_id]
            << "\n";

        std::cout
            << "  SCORE    = "
            << d.score
            << "\n";

        std::cout
            << "  BOX      = "
            << "x=" << d.x
            << " y=" << d.y
            << " w=" << d.width
            << " h=" << d.height
            << "\n";
    }

    std::cout
        << "\n========================================\n"
        << " TIMING SUMMARY\n"
        << "========================================\n"
        << "CPU Sobel            = "
        << cpu_sobel_ms
        << " ms\n"
        << "HLS Sobel            = "
        << hls_sobel_ms
        << " ms\n"
        << "Sobel speedup        = "
        << ((hls_sobel_ms > 0.0)
                ? (cpu_sobel_ms / hls_sobel_ms)
                : 0.0)
        << " x\n"
        << "----------------------------------------\n"
        << "YOLO preprocessing   = "
        << preprocess_ms
        << " ms\n"
        << "DPU execution        = "
        << dpu_ms
        << " ms\n"
        << "YOLO postprocessing  = "
        << postprocess_ms
        << " ms\n"
        << "Sequential compute sum = "
        << sequential_compute_ms
        << " ms\n"
        << "Estimated compute FPS  = "
        << ((sequential_compute_ms > 0.0)
                ? (1000.0 / sequential_compute_ms)
                : 0.0)
        << "\n"
        << "========================================\n";

    /*
     * Append benchmark measurements to CSV.
     *
     * The file is written in the current working directory.
     * Image loading, JPEG saving and visualization are not part
     * of the measured compute timing.
     */
    const std::string benchmark_csv =
        "benchmark_results.csv";

    bool write_csv_header = false;

    {
        std::ifstream existing_csv(
            benchmark_csv,
            std::ios::binary |
            std::ios::ate);

        write_csv_header =
            !existing_csv.good() ||
            existing_csv.tellg() == 0;
    }

    std::ofstream csv(
        benchmark_csv,
        std::ios::app);

    if (!csv) {
        std::cerr
            << "WARNING: Could not open "
            << benchmark_csv
            << " for writing\n";
    } else {

        if (write_csv_header) {
            csv
                << "image,"
                << "width,"
                << "height,"
                << "cpu_sobel_ms,"
                << "hls_sobel_ms,"
                << "sobel_speedup,"
                << "yolo_preprocess_ms,"
                << "dpu_ms,"
                << "yolo_postprocess_ms,"
                << "sequential_compute_ms,"
                << "estimated_compute_fps,"
                << "detections\n";
        }

        const double csv_sobel_speedup =
            (hls_sobel_ms > 0.0)
                ? (cpu_sobel_ms /
                   hls_sobel_ms)
                : 0.0;

        const double csv_compute_fps =
            (sequential_compute_ms > 0.0)
                ? (1000.0 /
                   sequential_compute_ms)
                : 0.0;

        csv
            << std::fixed
            << std::setprecision(6)
            << image_path << ","
            << image.cols << ","
            << image.rows << ","
            << cpu_sobel_ms << ","
            << hls_sobel_ms << ","
            << csv_sobel_speedup << ","
            << preprocess_ms << ","
            << dpu_ms << ","
            << postprocess_ms << ","
            << sequential_compute_ms << ","
            << csv_compute_fps << ","
            << detections.size()
            << "\n";

        csv.close();

        std::cout
            << "Benchmark CSV updated = "
            << benchmark_csv
            << "\n";
    }

    /*
     * Draw detections on the original image.
     *
     * Because this model path uses direct resize rather than
     * letterbox, normalized coordinates can be scaled directly
     * back to the original image width and height.
     */
    cv::Mat result_image = image.clone();

    for (const auto& d : detections) {

        int x1 = static_cast<int>(
            d.x * static_cast<float>(image.cols));

        int y1 = static_cast<int>(
            d.y * static_cast<float>(image.rows));

        int x2 = static_cast<int>(
            (d.x + d.width) *
            static_cast<float>(image.cols));

        int y2 = static_cast<int>(
            (d.y + d.height) *
            static_cast<float>(image.rows));

        x1 = std::max(0, std::min(x1, image.cols - 1));
        y1 = std::max(0, std::min(y1, image.rows - 1));
        x2 = std::max(0, std::min(x2, image.cols - 1));
        y2 = std::max(0, std::min(y2, image.rows - 1));

        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        cv::rectangle(
            result_image,
            cv::Point(x1, y1),
            cv::Point(x2, y2),
            cv::Scalar(0, 255, 0),
            2);

        const std::string label =
            std::string(COCO_CLASSES[d.class_id]) +
            " " +
            std::to_string(d.score).substr(0, 4);

        int baseline = 0;

        const cv::Size text_size =
            cv::getTextSize(
                label,
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
                &baseline);

        int text_y =
            std::max(y1, text_size.height + 4);

        cv::rectangle(
            result_image,
            cv::Point(
                x1,
                text_y - text_size.height - 4),
            cv::Point(
                std::min(
                    x1 + text_size.width + 4,
                    image.cols - 1),
                text_y + baseline),
            cv::Scalar(0, 255, 0),
            cv::FILLED);

        cv::putText(
            result_image,
            label,
            cv::Point(x1 + 2, text_y - 2),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 0),
            1);
    }

    const std::string output_image =
        "result.jpg";

    if (!cv::imwrite(
            output_image,
            result_image)) {

        std::cerr
            << "ERROR: Could not save "
            << output_image
            << "\n";

        return 1;
    }

    std::cout
        << "\nDetection image saved as: "
        << output_image
        << "\n";

    std::cout
        << "\n========================================\n"
        << " YOLOV5 POSTPROCESSING = OK\n"
        << "========================================\n";

    return 0;
}
