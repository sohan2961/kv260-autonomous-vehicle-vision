SUMMARY = "Recipe for  build an external sobel-dma-driver Linux kernel module"
SECTION = "PETALINUX/modules"
LICENSE = "GPLv2"
LIC_FILES_CHKSUM = "file://COPYING;md5=12f884d2ae1ff87c09e5b7ccc2c4ca7e"

inherit module

INHIBIT_PACKAGE_STRIP = "1"

SRC_URI = "file://Makefile \
           file://sobel-dma-driver.c \
	   file://COPYING \
          "

S = "${WORKDIR}"

# The inherit of module.bbclass will automatically name module packages with
# "kernel-module-" prefix as required by the oe-core build environment.

# Automatically load the Sobel platform driver at Linux boot.
# KERNEL_MODULE_AUTOLOAD disabled for FPGA-programming-before-driver-load

# Prevent udev/device-tree modalias from loading this driver
# before the FPGA programmable logic has been configured.
# Explicit "modprobe sobel-dma-driver" remains possible later.
KERNEL_MODULE_PROBECONF += "sobel-dma-driver"
module_conf_sobel-dma-driver = "blacklist sobel-dma-driver"
