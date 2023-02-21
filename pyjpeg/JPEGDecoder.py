from struct import unpack
# import numpy as np
from utils import ZIGZAG_SEQ
# import torch
marker_mapping = {
    0xFFD8: "Start of Image",
    0xFFE0: "Application Default Header",
    0xFFDB: "Quantization Table",
    0xFFC0: "Start of Frame",
    0xFFC4: "Define Huffman Table",
    0xFFDA: "Start of Scan",
    0xFFD9: "End of Image"
}


class JPEGDecoder(object):
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
        self.Q_table = {}
        self.raw = None
        self.height = None
        self.width = None
        self.Q_table_ids = {}
        self.huff_table_ids = {}

    def decode(self):
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[:2])
            print(marker_mapping.get(marker))
            # Start of Image
            if marker == 0xFFD8:
                data = data[2:]
            # End of Image
            elif marker == 0xFFD9:
                return
            else:
                data = data[2:]
                chunk_len, = unpack(">H", data[:2])
                chunk = data[2:chunk_len]
                data = data[chunk_len:]
                # Quantization Table
                if marker == 0xFFDB:
                    self.get_Q_table(chunk)
                # Start of Frame
                elif marker == 0xFFC0:
                    self.SOF(chunk)
                # Define Huffman Table
                elif marker == 0xFFC4:
                    self.get_huffman_table(chunk)
                # Start of Scan
                elif marker == 0xFFDA:
                    used = self.scan(chunk, data)
                    data = data[used:]
                
    def get_Q_table(self, data):
        dest = data[0]
        table = unpack("B"*64, data[1:])
        self.Q_table[dest] = table

    def SOF(self, data):
        precision, self.height, self.width, components = unpack(">BHHB", data[:6])
        for i in range(components):
            id, samp, table_id = unpack("BBB", data[6 + i * 3: 9 + i * 3])
            self.Q_table_ids[id] = table_id

    def get_huffman_table(self, data):
        header = data[0]
        
        # class_ = header & 0x0F
        # dest = (header >> 4) & 0x0F
        lengths = unpack("B"*16, data[1:17])

        elements = []
        offset = 17
        for i in lengths:
            elements += unpack("B"*i, data[offset:offset+i])
            offset += i
        print(lengths, elements)
    def scan(self, chunk, data):
        components = chunk[0]
        for i in range(components):
            id, table_header = unpack("BB", chunk[1+2*i:3+2*i])
            self.huff_table_ids[id] = table_header
        
        assert data[-2:] == b'\xff\xd9'
        coeff = []
        index = 0
        print(data)
        print(self.Q_table_ids)
        print(self.huff_table_ids)
        input()
        # while True:
        #     if data[i] == 0xff:
        #         if data[i+1] == 0x00:
        #             i += 1
        #         else:
        #             break
        #     coeff.append(data[i])
        #     index += 1
        # print(coeff)
        return i
        
if __name__ == '__main__':
    img = JPEGDecoder(r'C:\Users\zhjc1124\Desktop\deeplabv3\pyjpeg\lena.jpg')
    img.decode()