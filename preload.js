const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Functions that can be invoked from the renderer
  selectFile: () => ipcRenderer.invoke('dialog:openFile'),
  loadConfig: () => ipcRenderer.invoke('load-config'),
  saveConfig: (configData) => ipcRenderer.invoke('save-config', configData),
  
  // Listener for receiving logs from the main process
  onLogMessage: (callback) => ipcRenderer.on('log-message', (_event, value) => callback(value)),
  
  // Function to remove the listener
  removeLogListener: () => ipcRenderer.removeAllListeners('log-message'),
  
  // Get system metrics
  getSystemMetrics: () => ipcRenderer.invoke('get-system-metrics'),
});
