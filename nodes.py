import os
import folder_paths
from .inference import MuseTalk_INFER
from .inference_realtime import Infer_Real_Time
from pydub import AudioSegment
from moviepy.editor import VideoFileClip,AudioFileClip

parent_directory = os.path.dirname(os.path.abspath(__file__))
input_path = folder_paths.get_input_directory()
out_path = folder_paths.get_output_directory()

class MuseTalkRealTime:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "audio":("AUDIO",),
                "video":("VIDEO",),
                "avatar_id":("STRING",{
                    "default": "talker1"
                }),
                "bbox_shift":("INT",{
                    "default":0
                }),
                "fps":("INT",{
                    "default":25
                }),
                "batch_size":("INT",{
                    "default":4
                }),
                "preparation":("BOOLEAN",{
                    "default":True
                })
            }
        }
    CATEGORY = "AIFSH_MuseTalk"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ("VIDEO",)

    OUTPUT_NODE = False

    FUNCTION = "process"

    def process(self,audio,video,avatar_id,bbox_shift,fps,batch_size,preparation):
        muse_talk_real_time = Infer_Real_Time()
        output_vid_name = muse_talk_real_time(audio, video,avatar_id,fps=fps,batch_size=batch_size,
                 preparation=preparation,bbox_shift=bbox_shift)
        return (output_vid_name,)


class MuseTalk:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "audio":("AUDIO",),
                "video":("VIDEO",),
                "bbox_shift":("INT",{
                    "default":0
                }),
                "fps":("INT",{
                    "default":25
                }),
                "batch_size":("INT",{
                    "default":8
                }),
                "batch_size_fa":("INT",{
                    "default":2
                }),
                "use_saved_coord":("BOOLEAN",{
                    "default":False
                })
            }
        }
    CATEGORY = "AIFSH_MuseTalk"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ("VIDEO",)

    OUTPUT_NODE = False

    FUNCTION = "process"

    def process(self,audio,video,bbox_shift,fps,batch_size,batch_size_fa,use_saved_coord):
        muse_talk = MuseTalk_INFER(bbox_shift,fps,batch_size,batch_size_fa,use_saved_coord)
        output_vid_name = muse_talk(video, audio)
        return (output_vid_name,)


class CombineAudioVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"vocal_AUDIO": ("AUDIO",),
                     "bgm_AUDIO": ("AUDIO",),
                     "video": ("VIDEO",)
                    }
                }

    CATEGORY = "AIFSH_MuseTalk"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ("VIDEO",)

    OUTPUT_NODE = False

    FUNCTION = "combine"

    def combine(self, vocal_AUDIO,bgm_AUDIO,video):
        vocal = AudioSegment.from_file(vocal_AUDIO)
        bgm = AudioSegment.from_file(bgm_AUDIO)
        audio = vocal.overlay(bgm)
        audio_file = os.path.join(out_path,"ip_lap_voice.wav")
        audio.export(audio_file, format="wav")
        cm_video_file = os.path.join(out_path,"voice_"+os.path.basename(video))
        video_clip = VideoFileClip(video)
        audio_clip = AudioFileClip(audio_file)
        new_video_clip = video_clip.set_audio(audio_clip)
        new_video_clip.write_videofile(cm_video_file)
        return (cm_video_file,)


class PreViewVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "video":("VIDEO",),
        }}
    
    CATEGORY = "AIFSH_MuseTalk"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_video"

    def load_video(self, video):
        video_name = os.path.basename(video)
        video_path_name = os.path.basename(os.path.dirname(video))
        return {"ui":{"video":[video_name,video_path_name]}}

class LoadVideo:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["mp4", "webm","mkv","avi"]]
        return {"required":{
            "video":(files,),
        }}
    
    CATEGORY = "AIFSH_MuseTalk"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ("VIDEO","AUDIO")

    OUTPUT_NODE = False

    FUNCTION = "load_video"

    def load_video(self, video):
        video_path = os.path.join(input_path,video)
        video_clip = VideoFileClip(video_path)
        audio_path = os.path.join(input_path,video+".wav")
        video_clip.audio.write_audiofile(audio_path)
        return (video_path,audio_path,)