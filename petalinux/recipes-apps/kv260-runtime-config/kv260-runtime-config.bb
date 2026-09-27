SUMMARY = "Permanent KV260 vision runtime configuration"
SECTION = "PETALINUX/apps"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

inherit systemd

SRC_URI = "file://10-eth0.network \
           file://kv260-vision.conf \
           file://start-vision-demo \
           file://kv260-vision-drivers.service \
          "

S = "${WORKDIR}"

SYSTEMD_SERVICE:${PN} = "kv260-vision-drivers.service"
SYSTEMD_AUTO_ENABLE:${PN} = "enable"

do_install() {
    install -d ${D}${sysconfdir}/systemd/network
    install -m 0644 10-eth0.network \
        ${D}${sysconfdir}/systemd/network/10-eth0.network

    install -d ${D}${sysconfdir}/modules-load.d
    install -m 0644 kv260-vision.conf \
        ${D}${sysconfdir}/modules-load.d/kv260-vision.conf

    install -d ${D}${bindir}
    install -m 0755 start-vision-demo \
        ${D}${bindir}/start-vision-demo

    install -d ${D}${systemd_system_unitdir}
    install -m 0644 kv260-vision-drivers.service \
        ${D}${systemd_system_unitdir}/kv260-vision-drivers.service
}

FILES:${PN} += " \
    ${sysconfdir}/systemd/network/10-eth0.network \
    ${sysconfdir}/modules-load.d/kv260-vision.conf \
    ${bindir}/start-vision-demo \
    ${systemd_system_unitdir}/kv260-vision-drivers.service \
"
