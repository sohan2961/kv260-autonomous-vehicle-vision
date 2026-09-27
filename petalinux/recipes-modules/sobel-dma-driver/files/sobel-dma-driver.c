// SPDX-License-Identifier: GPL-2.0
/*
 * KV260 HLS Sobel platform driver
 *
 * compatible = "xlnx,sobel-accel-1.0"
 *
 * Stage:
 *   - map AXI-Lite registers
 *   - obtain/request Sobel IRQ
 *   - configure 32-bit coherent DMA
 *   - allocate contiguous Sobel input/output DMA buffers
 *
 * Userspace mmap/ioctl interface will be added next.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/io.h>
#include <linux/interrupt.h>
#include <linux/dma-mapping.h>
#include <linux/slab.h>
#include <linux/mm.h>

#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/completion.h>
#include <linux/mutex.h>
#include <linux/uaccess.h>
#include <linux/jiffies.h>
#include <linux/ioctl.h>
/* --------------------------------------------------------- */
/* HLS Sobel AXI-Lite registers                              */
/* --------------------------------------------------------- */

#define SOBEL_REG_AP_CTRL       0x00
#define SOBEL_REG_GIE           0x04
#define SOBEL_REG_IER           0x08
#define SOBEL_REG_ISR           0x0c

#define SOBEL_REG_INPUT_LO      0x10
#define SOBEL_REG_INPUT_HI      0x14

#define SOBEL_REG_OUTPUT_LO     0x1c
#define SOBEL_REG_OUTPUT_HI     0x20

#define SOBEL_REG_WIDTH         0x28
#define SOBEL_REG_HEIGHT        0x30

/* AP_CTRL bits */
#define SOBEL_AP_START          BIT(0)
#define SOBEL_AP_DONE           BIT(1)
#define SOBEL_AP_IDLE           BIT(2)
#define SOBEL_AP_READY          BIT(3)
#define SOBEL_AUTO_RESTART      BIT(7)
#define SOBEL_INTERRUPT         BIT(9)

/* --------------------------------------------------------- */
/* Maximum Sobel image                                       */
/* --------------------------------------------------------- */

#define SOBEL_MAX_WIDTH         1920U
#define SOBEL_MAX_HEIGHT        1080U

/*
 * Sobel input/output are 8-bit grayscale images.
 *
 * 1920 * 1080 = 2,073,600 bytes.
 */
#define SOBEL_MAX_FRAME_BYTES \
((size_t)SOBEL_MAX_WIDTH * (size_t)SOBEL_MAX_HEIGHT)

/*
 * Keep each frame area page-aligned.
 *
 * Memory layout:
 *
 * dma_cpu
 *   |
 *   +-----------------------+
 *   | INPUT                 |
 *   | SOBEL_BUFFER_STRIDE   |
 *   +-----------------------+
 *   | OUTPUT                |
 *   | SOBEL_BUFFER_STRIDE   |
 *   +-----------------------+
 */
#define SOBEL_BUFFER_STRIDE \
PAGE_ALIGN(SOBEL_MAX_FRAME_BYTES)

#define SOBEL_TOTAL_DMA_BYTES \
(2UL * SOBEL_BUFFER_STRIDE)

/* --------------------------------------------------------- */


/* --------------------------------------------------------- */
/* Userspace Sobel run interface                             */
/* --------------------------------------------------------- */

struct sobel_run_request {
__u32 width;
__u32 height;
};

#define SOBEL_IOCTL_MAGIC 'S'

#define SOBEL_IOCTL_RUN \
_IOW(SOBEL_IOCTL_MAGIC, 0x01, struct sobel_run_request)

#define SOBEL_RUN_TIMEOUT_MS 5000U

/* --------------------------------------------------------- */

struct sobel_device {
struct device *dev;

void __iomem *regs;
int irq;

/*
 * One contiguous coherent DMA allocation containing
 * both input and output frame buffers.
 */
void *dma_cpu;
dma_addr_t dma_base;

void *input_cpu;
dma_addr_t input_dma;

void *output_cpu;
dma_addr_t output_dma;

size_t dma_size;


struct miscdevice miscdev;

struct completion run_done;
struct mutex run_lock;
};

/* --------------------------------------------------------- */


/* --------------------------------------------------------- */
/* Userspace DMA mmap interface                              */
/* --------------------------------------------------------- */

static int sobel_mmap(struct file *file,
      struct vm_area_struct *vma)
{
struct miscdevice *misc = file->private_data;
struct sobel_device *sobel;
unsigned long requested;

sobel = container_of(
misc,
struct sobel_device,
miscdev);

requested =
vma->vm_end - vma->vm_start;

/*
 * Userspace maps the complete coherent allocation:
 *
 * offset 0
 *
 * [ INPUT buffer ][ OUTPUT buffer ]
 */
if (vma->vm_pgoff != 0)
return -EINVAL;

if (requested != sobel->dma_size)
return -EINVAL;

return dma_mmap_coherent(
sobel->dev,
vma,
sobel->dma_cpu,
sobel->dma_base,
sobel->dma_size);
}


static long sobel_ioctl(struct file *file,
unsigned int cmd,
unsigned long arg)
{
struct miscdevice *misc = file->private_data;

struct sobel_device *sobel =
container_of(
misc,
struct sobel_device,
miscdev);

struct sobel_run_request request;

size_t frame_bytes;
unsigned long timeout;
u32 stale_isr;
int ret = 0;

if (cmd != SOBEL_IOCTL_RUN)
return -ENOTTY;

if (copy_from_user(
&request,
(void __user *)arg,
sizeof(request)))
return -EFAULT;

if (request.width == 0 ||
    request.height == 0 ||
    request.width > SOBEL_MAX_WIDTH ||
    request.height > SOBEL_MAX_HEIGHT)
return -EINVAL;

frame_bytes =
(size_t)request.width *
(size_t)request.height;

if (frame_bytes > SOBEL_BUFFER_STRIDE)
return -EINVAL;

/*
 * The input/output DMA buffers are shared,
 * therefore permit only one hardware run at a time.
 */
if (mutex_lock_interruptible(
&sobel->run_lock))
return -ERESTARTSYS;

reinit_completion(
&sobel->run_done);

/*
 * Disable interrupts while configuring hardware.
 */
writel(
0,
sobel->regs + SOBEL_REG_GIE);

writel(
0,
sobel->regs + SOBEL_REG_IER);

/*
 * Clear stale HLS interrupt state.
 * ISR is Toggle-On-Write.
 */
stale_isr =
readl(
sobel->regs +
SOBEL_REG_ISR);

if (stale_isr)
writel(
stale_isr,
sobel->regs +
SOBEL_REG_ISR);

/*
 * Input physical/DMA address.
 */
writel(
lower_32_bits(
sobel->input_dma),
sobel->regs +
SOBEL_REG_INPUT_LO);

writel(
upper_32_bits(
sobel->input_dma),
sobel->regs +
SOBEL_REG_INPUT_HI);

/*
 * Output physical/DMA address.
 */
writel(
lower_32_bits(
sobel->output_dma),
sobel->regs +
SOBEL_REG_OUTPUT_LO);

writel(
upper_32_bits(
sobel->output_dma),
sobel->regs +
SOBEL_REG_OUTPUT_HI);

/*
 * Image dimensions.
 */
writel(
request.width,
sobel->regs +
SOBEL_REG_WIDTH);

writel(
request.height,
sobel->regs +
SOBEL_REG_HEIGHT);

/*
 * Enable ap_done interrupt.
 */
writel(
BIT(0),
sobel->regs +
SOBEL_REG_IER);

writel(
BIT(0),
sobel->regs +
SOBEL_REG_GIE);

/*
 * Start HLS Sobel.
 */
writel(
SOBEL_AP_START,
sobel->regs +
SOBEL_REG_AP_CTRL);

timeout =
wait_for_completion_timeout(
&sobel->run_done,
msecs_to_jiffies(
SOBEL_RUN_TIMEOUT_MS));

/*
 * Disable interrupts after transaction.
 */
writel(
0,
sobel->regs +
SOBEL_REG_GIE);

writel(
0,
sobel->regs +
SOBEL_REG_IER);

if (timeout == 0) {
dev_err(
sobel->dev,
"Sobel timeout: %ux%u AP_CTRL=0x%08x\n",
request.width,
request.height,
readl(
sobel->regs +
SOBEL_REG_AP_CTRL));

ret = -ETIMEDOUT;
goto out_unlock;
}

dev_dbg(
sobel->dev,
"Sobel complete: %ux%u (%zu bytes)\n",
request.width,
request.height,
frame_bytes);

out_unlock:

mutex_unlock(
&sobel->run_lock);

return ret;
}

/* --------------------------------------------------------- */

static const struct file_operations sobel_fops = {
.owner  = THIS_MODULE,
.mmap   = sobel_mmap,
.unlocked_ioctl = sobel_ioctl,
.llseek = no_llseek,
};

/* --------------------------------------------------------- */

static irqreturn_t sobel_irq_handler(int irq, void *data)
{
struct sobel_device *sobel = data;
u32 status;

status =
readl(
sobel->regs +
SOBEL_REG_ISR);

if (!status)
return IRQ_NONE;

/*
 * HLS ISR is Toggle-On-Write.
 */
writel(
status,
sobel->regs +
SOBEL_REG_ISR);

/*
 * ISR bit 0 = ap_done.
 */
if (status & BIT(0))
complete(
&sobel->run_done);

dev_dbg(
sobel->dev,
"Sobel interrupt: ISR=0x%08x\n",
status);

return IRQ_HANDLED;
}

/* --------------------------------------------------------- */

static int sobel_probe(struct platform_device *pdev)
{
struct sobel_device *sobel;
int ret;

dev_info(&pdev->dev,
 "probing KV260 HLS Sobel accelerator\n");

sobel = devm_kzalloc(&pdev->dev,
     sizeof(*sobel),
     GFP_KERNEL);

if (!sobel)
return -ENOMEM;

sobel->dev = &pdev->dev;

init_completion(
&sobel->run_done);

mutex_init(
&sobel->run_lock);

/*
 * Map:
 *
 * reg = <0x00 0xa0000000 0x00 0x10000>;
 */
sobel->regs =
devm_platform_ioremap_resource(pdev, 0);

if (IS_ERR(sobel->regs))
return PTR_ERR(sobel->regs);

/*
 * Keep the coherent Sobel DMA allocation below 4 GiB.
 *
 * The generated HLS pointer registers are 64-bit, but
 * using low DDR simplifies the first hardware integration.
 */
ret = dma_set_mask_and_coherent(
&pdev->dev,
DMA_BIT_MASK(32));

if (ret) {
dev_err(&pdev->dev,
"unable to configure 32-bit DMA mask\n");
return ret;
}

/* Obtain Sobel interrupt from device tree. */
sobel->irq =
platform_get_irq(pdev, 0);

if (sobel->irq < 0)
return sobel->irq;

ret = devm_request_irq(
&pdev->dev,
sobel->irq,
sobel_irq_handler,
0,
dev_name(&pdev->dev),
sobel);

if (ret) {
dev_err(&pdev->dev,
"failed to request IRQ %d: %d\n",
sobel->irq,
ret);
return ret;
}

/*
 * Disable HLS interrupts until the RUN interface
 * is implemented.
 */
writel(0x0,
       sobel->regs + SOBEL_REG_GIE);

writel(0x0,
       sobel->regs + SOBEL_REG_IER);

/* -------------------------------------------------- */
/* Allocate coherent DDR for Sobel                    */
/* -------------------------------------------------- */

sobel->dma_size =
SOBEL_TOTAL_DMA_BYTES;

sobel->dma_cpu =
dma_alloc_coherent(
&pdev->dev,
sobel->dma_size,
&sobel->dma_base,
GFP_KERNEL);

if (!sobel->dma_cpu) {
dev_err(&pdev->dev,
"failed to allocate %zu bytes of coherent DMA memory\n",
sobel->dma_size);

return -ENOMEM;
}

/*
 * Split the single allocation into two page-aligned
 * frame areas.
 */
sobel->input_cpu =
sobel->dma_cpu;

sobel->input_dma =
sobel->dma_base;

sobel->output_cpu =
(void *)((u8 *)sobel->dma_cpu +
 SOBEL_BUFFER_STRIDE);

sobel->output_dma =
sobel->dma_base +
SOBEL_BUFFER_STRIDE;

/*
 * Start both frame buffers cleared.
 */
memset(sobel->dma_cpu,
       0,
       sobel->dma_size);


/* Register /dev/sobel_accel */

sobel->miscdev.minor =
MISC_DYNAMIC_MINOR;

sobel->miscdev.name =
"sobel_accel";

sobel->miscdev.fops =
&sobel_fops;

sobel->miscdev.parent =
&pdev->dev;

ret = misc_register(
&sobel->miscdev);

if (ret) {
dev_err(
&pdev->dev,
"failed to register /dev/sobel_accel: %d\n",
ret);

dma_free_coherent(
&pdev->dev,
sobel->dma_size,
sobel->dma_cpu,
sobel->dma_base);

sobel->dma_cpu = NULL;

return ret;
}

platform_set_drvdata(
pdev,
sobel);

dev_info(&pdev->dev,
 "Sobel DMA allocated: total=%zu bytes\n",
 sobel->dma_size);

dev_info(&pdev->dev,
 "input DMA=%pad, output DMA=%pad\n",
 &sobel->input_dma,
 &sobel->output_dma);

dev_info(&pdev->dev,
 "max frame=%ux%u (%zu bytes)\n",
 SOBEL_MAX_WIDTH,
 SOBEL_MAX_HEIGHT,
 SOBEL_MAX_FRAME_BYTES);

dev_info(&pdev->dev,
 "Sobel driver ready: IRQ=%d AP_CTRL=0x%08x\n",
 sobel->irq,
 readl(sobel->regs +
       SOBEL_REG_AP_CTRL));

return 0;
}

/* --------------------------------------------------------- */

static int sobel_remove(struct platform_device *pdev)
{
struct sobel_device *sobel =
platform_get_drvdata(pdev);

if (!sobel)
return 0;

/*
 * Remove the userspace device before freeing
 * its coherent DMA backing memory.
 */
misc_deregister(
&sobel->miscdev);


/*
 * Disable HLS interrupts.
 */
writel(0x0,
       sobel->regs + SOBEL_REG_GIE);

writel(0x0,
       sobel->regs + SOBEL_REG_IER);

if (sobel->dma_cpu) {
dma_free_coherent(
&pdev->dev,
sobel->dma_size,
sobel->dma_cpu,
sobel->dma_base);

sobel->dma_cpu = NULL;
}

dev_info(&pdev->dev,
 "KV260 HLS Sobel driver removed\n");

return 0;
}

/* --------------------------------------------------------- */

static const struct of_device_id
sobel_of_match[] = {
{
.compatible =
"xlnx,sobel-accel-1.0",
},
{ }
};

MODULE_DEVICE_TABLE(
of,
sobel_of_match);

/* --------------------------------------------------------- */

static struct platform_driver
sobel_platform_driver = {
.probe  = sobel_probe,
.remove = sobel_remove,

.driver = {
.name =
"sobel-dma-driver",

.of_match_table =
sobel_of_match,
},
};

module_platform_driver(
sobel_platform_driver);

/* --------------------------------------------------------- */

MODULE_AUTHOR("Md Mostafizur Rahman");
MODULE_DESCRIPTION("KV260 HLS Sobel coherent DMA platform driver");
MODULE_LICENSE("GPL");
MODULE_VERSION("1.3");
