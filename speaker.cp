#include "AudioTools.h"
#include "BluetoothA2DPSink.h"

I2SStream i2s;
BluetoothA2DPSink a2dp_sink(i2s);

void setup() {
    Serial.begin(115200);
    
    auto cfg = i2s.defaultConfig();
    cfg.pin_bck = 26;      // BCLK
    cfg.pin_ws = 25;       // LRC
    cfg.pin_data = 22;     // DIN
    i2s.begin(cfg);
    
    a2dp_sink.start("HoodedMaker Speaker");
    
    Serial.println("Bluetooth колонка готова!");
    Serial.println("Подключитесь к HoodedMaker Speaker");
}

void loop() {
    delay(1000);
}
