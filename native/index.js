const { getVRAMInfo } = require('./build/Release/metal_vram.node');

module.exports = {
  getVRAMInfo: () => {
    try {
      return getVRAMInfo();
    } catch (error) {
      console.error('[Metal VRAM] Error:', error);
      return { error: error.message, total: 0, used: 0 };
    }
  }
};
