FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}:"

SRC_URI:append = " file://bsp.cfg file://0001-xlnx-dpu-v4.1-fix-zero-upper-fingerprint.patch"
KERNEL_FEATURES:append = " bsp.cfg"

# DPUCZDX8G Vivado-flow driver:
# build as module but prevent automatic probe before FPGA programming.
KERNEL_MODULE_PROBECONF += "xlnx_dpu"
module_conf_xlnx_dpu = "blacklist xlnx_dpu"
