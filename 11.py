import paddle
import argparse
import cv2
import numpy as np
import os
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel

from face_detector import *  # YoloDetector 포함 (yolov5n_state_dict.pt)
from PIL import Image
import io
import glob
# from google.colab.patches import cv2_imshow
from facenet_pytorch import InceptionResnetV1
import torch
import torch.nn as nn
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

'''
def get_id_emb(id_net, id_img_path):
    id_img = cv2.imread(id_img_path)

    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2paddle(id_img)
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std

    id_emb, id_feature = id_net(id_img)
    id_emb = l2_norm(id_emb)

    return id_emb, id_feature
'''


def get_id_emb_from_image(id_net, id_img):
    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2paddle(id_img)
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std
    id_emb, id_feature = id_net(id_img)
    id_emb = l2_norm(id_emb)

    return id_emb, id_feature


# -> get_id_emb_from_image
def image_test_multi_face(args, source_aligned_images, target_aligned_images):
    #paddle.set_device("gpu" if args.use_gpu else 'cpu')
    paddle.set_device("gpu" if args.use_gpu else 'cpu')
    faceswap_model = FaceSwap(args.use_gpu)

    id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
    id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))

    id_net.eval()

    weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')

    #target_path = args.target_img_path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')

    start_idx = args.target_img_path.rfind('/')
    if start_idx > 0:
        target_name = args.target_img_path[args.target_img_path.rfind('/'):]
    else:
        target_name = args.target_img_path
    origin_att_img = cv2.imread(args.target_img_path)
    #id_emb, id_feature = get_id_emb(id_net, base_path + '_aligned.png')

    for idx, target_aligned_image in enumerate(target_aligned_images):
        id_emb, id_feature = get_id_emb_from_image(
            id_net, source_aligned_images[idx % len(source_aligned_images)][0])

        # target_aligned_image에 대해 코드 돌려서 확인
        faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
        faceswap_model.eval()
        # print(target_aligned_image.shape)

        att_img = cv2paddle(target_aligned_image[0])
        #import time
        #start = time.perf_counter()

        res, mask = faceswap_model(att_img)
        #print('process time :{}', time.perf_counter() - start)
        res = paddle2cv(res)

        # dest[landmarks[idx][0]:landmarks[idx][1],:] =

        back_matrix = target_aligned_images[idx %
                                            len(target_aligned_images)][1]
        mask = np.transpose(mask[0].numpy(), (1, 2, 0))
        origin_att_img = dealign(res, origin_att_img, back_matrix, mask)
        '''
        if args.merge_result:
            back_matrix = np.load(base_path + '_back.npy')
            mask = np.transpose(mask[0].numpy(), (1, 2, 0))
            res = dealign(res, origin_att_img, back_matrix, mask)
            '''
    cv2.imwrite(os.path.join(args.output_dir, os.path.basename(
        target_name.format(idx))), origin_att_img)


'''
def face_align(landmarkModel, image_path, merge_result=False, image_size=224): 
    if os.path.isfile(image_path):
        img_list = [image_path]
    else:
        img_list = [os.path.join(image_path, x) for x in os.listdir(image_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
    for path in img_list:
        img = cv2.imread(path)
        landmark = landmarkModel.get(img)
        if landmark is not None:
            base_path = path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
            aligned_img, back_matrix = align_img(img, landmark, image_size)
            # np.save(base_path + '.npy', landmark)
            cv2.imwrite(base_path + '_aligned.png', aligned_img)
            if merge_result:
                np.save(base_path + '_back.npy', back_matrix)
'''

###

yolo = YoloDetector(target_size=720, device ="cuda:0", min_face = 50)

resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()


class Tuning(nn.Module):
    def __init__(self):
        super(Tuning, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = resnet(x)
        x = self.classifier(x)
        return x


path = "/content/drive/MyDrive/DL_Project/SVM/DLpj/model_for_inference/model_v1.pt"
chanhyeong = torch.load(path, map_location=device)
chanhyeong.eval()  # 평가 모드로 모델 불러오기

###

def faces_align(landmarkModel, image_path, image_size=224):

    aligned_imgs = []
    if os.path.isfile(image_path):
        img_list = [image_path]
    else:
        img_list = [os.path.join(image_path, x) for x in os.listdir(
            image_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
    for path in img_list:
        img = cv2.imread(path)

        aligned_imgs = []
        output_pb_list = []

        landmarks = landmarkModel.gets(img)
        for landmark in landmarks:
            if landmark is not None:
                aligned_img, back_matrix = align_img(img, landmark, image_size)

                ###
                
                bboxes, points = yolo.predict(aligned_img)
                # crop and align image
                face = yolo.align(aligned_img, points[0])

                print(type(face))
                print(face.shape)
                
                data_transform = transforms.Compose(
                    [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                
                if torch.tensor(face).dim == 3:
                    face = torch.tensor(face) / 255.0  # face는 3차원 tensor가 된다
                    face = face.permute(2, 0, 1)
                    face = face.float()
                    face = data_transform(face)
                    face = face.unsqueeze(0)  # face는 4차원이 된다
                    # face = face.float()
                    face = face.to(device)

                    output_pb = chanhyeong(face)[0]
                    output_pb_list.append(output_pb)

                ###

                aligned_imgs.append([aligned_img, back_matrix])

        ###

        max_index = output_pb_list.index(max(output_pb_list))
        aligned_imgs.pop(max_index)

        ###

    return aligned_imgs


if __name__ == '__main__':  # -> faces_align, image_test_multi_face

    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")
    parser.add_argument('--source_img_path', type=str,
                        help='path to the source image')
    parser.add_argument('--target_img_path', type=str,
                        help='path to the target images')
    parser.add_argument('--output_dir', type=str,
                        default='results', help='path to the output dirs')
    parser.add_argument('--image_size', type=int, default=224,
                        help='size of the test images (224 SimSwap | 256 FaceShifter)')
    parser.add_argument('--merge_result', type=bool,
                        default=True, help='output with whole image')
    parser.add_argument('--need_align', type=bool,
                        default=True, help='need to align the image')
    parser.add_argument('--use_gpu', type=bool, default=False)

    args = parser.parse_args()
    if args.need_align:
        landmarkModel = LandmarkModel(name='landmarks')
        landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
        source_aligned_images = faces_align(
            landmarkModel, args.source_img_path)
        target_aligned_images = faces_align(
            landmarkModel, args.target_img_path, args.image_size)
    os.makedirs(args.output_dir, exist_ok=True)
    image_test_multi_face(args, source_aligned_images, target_aligned_images)
