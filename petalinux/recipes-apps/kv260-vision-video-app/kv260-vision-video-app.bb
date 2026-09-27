#
# Separate KV260 video/live perception application.
#

SUMMARY = "KV260 HLS Sobel and Vitis AI DPU video perception application"
SECTION = "PETALINUX/apps"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = "file://kv260-vision-video-app.cpp \
           file://Makefile \
          "

S = "${WORKDIR}"

do_compile() {
        oe_runmake
}

do_install() {
        install -d ${D}${bindir}
        install -m 0755 kv260-vision-video-app ${D}${bindir}
}

DEPENDS += "xir unilog vart opencv"
RDEPENDS:${PN} += "xir unilog vart"
