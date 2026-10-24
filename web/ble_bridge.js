(() => {
  let device = null;
  let characteristic = null;

  window.esp32Ble = {
    async connect(serviceUuid, characteristicUuid, onData, onStatus) {
      try {
        onStatus("Opening Bluetooth device chooser...");

        device = await navigator.bluetooth.requestDevice({
          filters: [{ name: "ESP32-WROOM-32 Sound Alert" }],
          optionalServices: [serviceUuid]
        });

        onStatus("Connecting to " + (device.name || "ESP32") + "...");

        device.addEventListener("gattserverdisconnected", () => {
          characteristic = null;
          onStatus("ESP32 disconnected.");
        });

        const server = await device.gatt.connect();

        const service = await server.getPrimaryService(serviceUuid);

        characteristic =
          await service.getCharacteristic(characteristicUuid);

        characteristic.addEventListener(
          "characteristicvaluechanged",
          event => {
            const value = event.target.value;
            const bytes = [];

            for (let i = 0; i < value.byteLength; i++) {
              bytes.push(value.getUint8(i));
            }

            onData(JSON.stringify(bytes));
          }
        );

        await characteristic.startNotifications();

        try {
          const value = await characteristic.readValue();
          const bytes = [];

          for (let i = 0; i < value.byteLength; i++) {
            bytes.push(value.getUint8(i));
          }

          if (bytes.length > 0) {
            onData(JSON.stringify(bytes));
          }
        } catch (_) {
          // Notifications can work even if the initial read fails.
        }

        onStatus("Connected to " + (device.name || "ESP32"));
        return true;
      } catch (error) {
        onStatus("BLE error: " + error.message);
        return false;
      }
    },

    async disconnect() {
      if (device && device.gatt && device.gatt.connected) {
        device.gatt.disconnect();
      }

      characteristic = null;
      device = null;
    },

    get deviceName() {
      return device?.name || "No Device";
    },

    get isConnected() {
      return !!(device && device.gatt && device.gatt.connected);
    }
  };
})();