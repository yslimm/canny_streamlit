import cv2
import socket
import numpy as np


class Lighting():
    def __init__(self):
        target_ip = '192.168.0.30'
        target_port = 1000
        buffer_size = 4096

        # 1.ソケットオブジェクトの作成
        self.tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 2.サーバに接続
        self.tcp_client.connect((target_ip, target_port))

        # 3. messag3e
        print('TCP connected.')

    def make_com_bright(self, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8):
        com_bright = 'W11'
        c_ch1 = str(ch1).zfill(4)
        c_ch2 = str(ch2).zfill(4)
        c_ch3 = str(ch3).zfill(4)
        c_ch4 = str(ch4).zfill(4)
        c_ch5 = str(ch5).zfill(4)
        c_ch6 = str(ch6).zfill(4)
        c_ch7 = str(ch7).zfill(4)
        c_ch8 = str(ch8).zfill(4)
        command = com_bright + '01' + c_ch1 + '02' + c_ch2 + '03' + c_ch3 + '04' + c_ch4 + '05' \
                  + c_ch5 + '06' + c_ch6 + '07' + c_ch7 + '08' + c_ch8
        return command.encode()

    def control_light(self, num, bright):
        br_t = int(bright / 2)
        i = num

        if i == 0:
            self.tcp_client.send(self.make_com_bright(bright, 0, 0, 0, 0, 0, 0, 0))

        elif i == 1:
            self.tcp_client.send(self.make_com_bright(0, bright, 0, 0, 0, 0, 0, 0))

        elif i == 2:
            self.tcp_client.send(self.make_com_bright(0, 0, bright, 0, 0, 0, 0, 0))

        elif i == 3:
            self.tcp_client.send(self.make_com_bright(0, 0, 0, bright, 0, 0, 0, 0))

        elif i == 4:
            self.tcp_client.send(self.make_com_bright(0, 0, 0, 0, bright, 0, 0, 0))

        elif i == 5:
            self.tcp_client.send(self.make_com_bright(0, 0, 0, 0, 0, bright, 0, 0))

        elif i == 6:
            self.tcp_client.send(self.make_com_bright(0, 0, 0, 0, 0, 0, bright, 0))

        else:
            self.tcp_client.send(self.make_com_bright(0, 0, 0, 0, 0, 0, 0, bright))


    def turn_off(self):
        self.tcp_client.send(self.make_com_bright(0, 0, 0, 0, 0, 0, 0, 0))

    def turn_on(self, br):
        val_br = br
        self.tcp_client.send(self.make_com_bright(val_br, val_br, val_br, val_br, val_br, val_br, val_br, val_br))



