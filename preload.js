const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  selectFile: () => ipcRenderer.invoke('dialog:openFile'),
  loadConfig: () => ipcRenderer.invoke('load-config'),
  saveConfig: (configData) => ipcRenderer.invoke('save-config', configData),
});
