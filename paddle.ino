/*
 *  This sketch sends a message to a TCP server
 *
 */
#include <WiFi.h>
#include <WiFiMulti.h>

#include <string>

WiFiMulti WiFiMulti;


#include <Adafruit_LSM6DSOX.h>
Adafruit_LSM6DSOX lsm6ds;

#include <Adafruit_LIS3MDL.h>
Adafruit_LIS3MDL lis3mdl;

const uint16_t port = 60321;
const char * host = "71.244.253.230 ";

void calibrate(WiFiClient);
void data_send(float, float, WiFiClient);
void wait_for_response(WiFiClient);
void get_instructions(WiFiClient);
void setup_sensors();
void get_status(WiFiClient);
void connect();

void setup() {
  Serial.begin(19200);
}

void loop() {
  connect();
}

void connect() {
  WiFiMulti.addAP("ESP", "obpq6678");

  if (WiFiMulti.run() != WL_CONNECTED) {
    Serial.println("Waiting for WiFi... ");
    return;
  }

  WiFiClient client;

  if (!client.connect(host, port)) {
    Serial.println("Connection failed.");
    Serial.println("Waiting 5 seconds before retrying...");
    delay(5000);
    connect();
  }
  else {
    Serial.println("Connected to server!");
    get_status(client); 
  }
}

void get_status(WiFiClient client) {
  bool lsm6ds_success, lis3mdl_success;
  lsm6ds_success = lsm6ds.begin_I2C();
  lis3mdl_success = lis3mdl.begin_I2C();

  if (!lsm6ds_success or !lis3mdl_success) {
    if (!(lsm6ds_success && !lis3mdl_success)) {
      client.print("No sensors found try recharging!");
      Serial.println("No sensors found try recharging!");
    }
    else {
      client.print("Failed to find LSM6DS chip. Something might be broken!");
      Serial.println("Failed to find LSM6DS chip. Something might be broken!");
    }
  }


  if (lsm6ds_success && lis3mdl_success) {
    client.print("LSM6DS and LIS3MDL Found! Waiting for commands.");
    Serial.println("LSM6DS and LIS3MDL Found! Waiting for commands.");
    setup_sensors();
    get_instructions(client);
  }
  Serial.println("Connection Lost");
}

void setup_sensors() {
  lsm6ds.setAccelRange(LSM6DS_ACCEL_RANGE_2_G);
  lsm6ds.setAccelDataRate(LSM6DS_RATE_12_5_HZ);
  lsm6ds.setGyroRange(LSM6DS_GYRO_RANGE_250_DPS);
  lsm6ds.setGyroDataRate(LSM6DS_RATE_12_5_HZ);
  lis3mdl.setDataRate(LIS3MDL_DATARATE_155_HZ);
  lis3mdl.setRange(LIS3MDL_RANGE_4_GAUSS);
  lis3mdl.setPerformanceMode(LIS3MDL_MEDIUMMODE);
  lis3mdl.setOperationMode(LIS3MDL_CONTINUOUSMODE);
  lis3mdl.setIntThreshold(500);
  lis3mdl.configInterrupt(false, false, true, // enable z axis
                          true, // polarity
                          false, // don't latch
                          true); // enabled!

}

void get_instructions(WiFiClient client) {
  float time, freq;
  wait_for_response(client);
  if (client.available()) {
    time = atof(client.readStringUntil('/r').c_str());
    client.print("Recieved Time");
    Serial.print("time:");
    Serial.println(time);
  }
  delay (10);
  wait_for_response(client);

  if (client.available()) {
    freq = atof(client.readStringUntil('/r').c_str());
    Serial.print("freq:");
    Serial.println(freq);
    if (freq == 0) {
      Serial.print("Exited because of divide by 0");
      connect();
    }
  }

  data_send(time, freq, client);
  delay(2000);
  client.print("Ready for next instructions!");
  Serial.println("Ready for next instructions!");
  get_instructions(client);
}

void wait_for_response(WiFiClient client) {
  int maxloops = 0;
  while (!client.available() && maxloops < 1000){
    maxloops++;
    delay(100);
  }
  if (maxloops == 1000) {
    client.print("Connection timed out.");
    Serial.println("Connection timed out.");
    connect();
  }
}

void data_send(float time, float freq, WiFiClient client) {
 /* if (calibrate == 1) {
    client.print("Calibrate mode activated.");
    Serial.println("Calibrate mode activated.");
  }*/
  float count;
  count = time;
  while (count > 0) {

    Serial.println("Sending Data");
    //if (!client.connect(host, port)) {
    //  Serial.println("Connection failed.");
    //  return;
    //}
    sensors_event_t accel, gyro, mag, temp;

    lsm6ds.getEvent(&accel, &gyro, &temp);
    lis3mdl.getEvent(&mag);
 
      
    client.print(time - count);
    
    count = count - (1.0/freq);
    Serial.println(count);
    Serial.println(time);
    
   
    client.print(",");
    client.print(accel.acceleration.x);
    client.print(",");
    client.print(accel.acceleration.y);
    client.print(",");
    client.print(accel.acceleration.z);
    client.print(",");
    client.print(gyro.gyro.x);
    client.print(",");
    client.print(gyro.gyro.y);
    client.print(",");
    client.print(gyro.gyro.z);
    client.print(",");
    client.print(mag.magnetic.x);
    client.print(",");
    client.print(mag.magnetic.y);
    client.print(",");
    client.print(mag.magnetic.z);
    client.print("*");

    delay(1000/freq);
  }
} 



