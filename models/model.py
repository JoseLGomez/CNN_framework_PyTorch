import torch
import os
import numpy as np
import urllib
import wget
import io
import httplib2
import apiclient
from apiclient.http import MediaIoBaseDownload
from apiclient.discovery import build
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.url = None

    def forward(self, x):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = self.get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
    def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        """Make a 2D bilinear kernel suitable for upsampling"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                          dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight).float()

    def load_basic_weights(self, basic_model_path, net_name):
        if not os.path.exists(basic_model_path):
            os.makedirs(basic_model_path)
        filename = os.path.join(basic_model_path, 'basic_'+ net_name +'.pth')
        self.download_if_not_exist(filename)
        input()
        self.restore_weights(filename)

    def download_if_not_exist(self, filename):
        # Download the file if it does not exist
        if not os.path.isfile(filename) and self.url is not None:
            urllib.urlretrieve(self.url, filename)
            #wget.download(self.url, filename)
            #self.download_google_drive2(self.url, filename)

    def restore_weights(self, filename):
        print('\t Loading basic model weights from ' + filename)
        self.load_state_dict(torch.load(os.path.join(filename)))

    '''def download_google_drive(self, url, outfile):
        http = httplib2.Http()
        drive_service = build('drive', 'v2', http=http)
        resp, content = drive_service._http.request(url)
        if resp.status == 200:
            if os.path.isfile(outfile):
                print "ERROR, %s already exist" % outfile
            else:
                with open(outfile, 'w') as f:
                    f.write(content)
                print "OK"
        else:
            print ('ERROR downloading')

    def download_google_drive2(self, url, outfile):
        id = url.split('=')
        print id[-1]
        print ('downloading file...')
        http = httplib2.Http()
        drive_service = build('drive', 'v2', http=http)
        fh = open(outfile, "w")
        request = drive_service.files().get_media(fileId=id[-1])
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print "Download %d%%." % int(status.progress() * 100)'''
