#
# This file is the kv260-vision-app recipe.
#

SUMMARY = "Simple kv260-vision-app application"
SECTION = "PETALINUX/apps"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = "file://kv260-vision-app.cpp \
           file://Makefile \
		  "

S = "${WORKDIR}"

do_compile() {
	     oe_runmake
}

do_install() {
	     install -d ${D}${bindir}
	     install -m 0755 kv260-vision-app ${D}${bindir}
}

DEPENDS += "xir unilog vart opencv"
RDEPENDS:${PN} += "xir unilog vart"
