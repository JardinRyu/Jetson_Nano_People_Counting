# Real-Time People Tracking and Counting on Jetson Nano
MobileNetV1-SSD based Person Detection + SORT Tracking

<p align="center">
  <img width="500" height="373" src="media/output.gif">
</p>

## Prerequisite

Jetson Nano - JetPack 4.5

## Install Dependencies

1. Clone this repository.

   ```shell
   $ git clone https://github.com/JardinRyu/Jetson_Nano_People_Counting
   $ cd Jetson_Nano_People_Counting/
   ```

2. Install requirements.

   ```shell
   # The installation process can take several hours. So be patient...
   # ${HOME}/Jetson_Nano_People_Counting/
   $ ./install.sh
   $ cd ssd/
   # ${HOME}/Jetson_Nano_People_Counting/ssd/
   $ ./install.sh
   $ ./build_engines.sh
   ```

## Usage
1. Run from USB Camera

    ```shell
    # CameraID = 0  
    $ python3 usbcam_tracking.py
    ```

2. Run from MIPI Camera (ex. Raspberry pi Cam v2)

    ```shell
    $ python3 mipicam_tracking.py
    ```

## References
- [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos)
- [sort](https://github.com/abewley/sort)
