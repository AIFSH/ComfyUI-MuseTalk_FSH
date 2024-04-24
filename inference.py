
import os
import cv2
import glob,copy
import shutil
from tqdm import tqdm
import torch
import pickle
import numpy as np
from cuda_malloc import cuda_malloc_supported
from typing import Any
import folder_paths
from .musetalk.utils.face_parsing import FaceParsing
from mmpose.apis import init_model
from .musetalk.utils.blending import get_image
from .musetalk.utils.utils import load_all_model,get_file_type,get_video_fps,datagen
from .musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder

parent_directory = os.path.dirname(os.path.abspath(__file__))
input_path = folder_paths.get_input_directory()
out_path = folder_paths.get_output_directory()

class MuseTalk_INFER:
    def __init__(self,bbox_shift=0,fps=25,
                 batch_size=8,batch_size_fa=2,
                 use_saved_coord=False) -> None:
        self.fps = fps
        self.bbox_shift = bbox_shift
        self.batch_size = batch_size
        self.batch_size_fa = batch_size_fa
        self.use_saved_coord = use_saved_coord
        self.device = torch.device("cuda" if cuda_malloc_supported() else "cpu")
        config_file = os.path.join(parent_directory,"musetalk","utils","dwpose","rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py")
        checkpoint_file = os.path.join(parent_directory,'models','dwpose','dw-ll_ucoco_384.pth')
        resnet_path = os.path.join(parent_directory,'models','face-parse-bisent','resnet18-5c106cde.pth')
        face_model_pth = os.path.join(parent_directory,"models','face-parse-bisent','79999_iter.pth")
        self.fp_model = FaceParsing(resnet_path,face_model_pth)
        self.dwpose_model = init_model(config_file, checkpoint_file, device=self.device)
        self.audio_processor,self.vae,self.unet,self.pe  = load_all_model(os.path.join(parent_directory,"models"))
        self.timesteps = torch.tensor([0], device=self.device)

    def __call__(self, video_path,audio_path,*args: Any, **kwds: Any) -> Any:
        input_basename = os.path.basename(video_path).split('.')[0]
        audio_basename  = os.path.basename(audio_path).split('.')[0]
        output_basename = f"{input_basename}_{audio_basename}"
        result_img_save_path = os.path.join(out_path,"musetalk_result", output_basename) # related to video & audio inputs
        os.makedirs(result_img_save_path, exist_ok=True)
        crop_coord_save_path = os.path.join(result_img_save_path, input_basename+".pkl") # only related to video input
        output_vid_name = os.path.join(out_path, output_basename+".mp4")
        
        ############################################## extract frames from source video ##############################################
        if get_file_type(video_path)=="video":
            save_dir_full = os.path.join(out_path,"musetalk_result",input_basename)
            os.makedirs(save_dir_full,exist_ok = True)
            png_path = os.path.join(save_dir_full,"%08d.png")
            cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {png_path}"
            os.system(cmd)
            input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
            fps = get_video_fps(video_path)
        else: # input img folder
            input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            fps = self.fps
        
        #print(input_img_list)
        ############################################## extract audio feature ##############################################
        whisper_feature = self.audio_processor.audio2feat(audio_path)
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        ############################################## preprocess input image  ##############################################
        if os.path.exists(crop_coord_save_path) and self.use_saved_coord:
            print("using extracted coordinates")
            with open(crop_coord_save_path,'rb') as f:
                coord_list = pickle.load(f)
            frame_list = read_imgs(input_img_list)
        else:
            print("extracting landmarks...time consuming")
            coord_list, frame_list = get_landmark_and_bbox(self.dwpose_model,input_img_list, self.batch_size_fa,self.bbox_shift)
            with open(crop_coord_save_path, 'wb') as f:
                pickle.dump(coord_list, f)
        
        i = 0
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
    
        # to smooth the first and the last frame
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        ############################################## inference batch by batch ##############################################
        print("start inference")
        video_num = len(whisper_chunks)
        batch_size = self.batch_size
        gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
        res_frame_list = []
        for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
            tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
            audio_feature_batch = torch.stack(tensor_list).to(self.unet.device) # torch, B, 5*N,384
            audio_feature_batch = self.pe(audio_feature_batch)
            
            pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)
                
        ############################################## pad to full image ##############################################
        print("pad talking image to original video")
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            bbox = coord_list_cycle[i%(len(coord_list_cycle))]
            ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
            except:
#                 print(bbox)
                continue
            
            combine_frame = get_image(self.fp_model, ori_frame,res_frame,bbox)
            cv2.imwrite(os.path.join(result_img_save_path,f"{str(i).zfill(8)}.png"),combine_frame)

        res_tmp_path = os.path.join(result_img_save_path,"%08d.png")
        cmd_img2video = f"ffmpeg -y -v fatal -r {fps} -f image2 -i {res_tmp_path} -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 {output_vid_name}"
        print(cmd_img2video)
        os.system(cmd_img2video)
        '''
        cmd_combine_audio = f"ffmpeg -y -v fatal -i {audio_path} -i temp.mp4 {output_vid_name}"
        print(cmd_combine_audio)
        os.system(cmd_combine_audio)
        '''
        del self.fp_model,self.dwpose_model,self.audio_processor,self.vae,self.unet,self.pe
        torch.cuda.empty_cache()
        # os.remove("temp.mp4")
        shutil.rmtree(result_img_save_path)
        print(f"result is save to {output_vid_name}")
        return output_vid_name
