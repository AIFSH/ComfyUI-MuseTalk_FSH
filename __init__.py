from .nodes import MuseTalk,LoadVideo,PreViewVideo,CombineAudioVideo,MuseTalkRealTime
WEB_DIRECTORY = "./web"
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "MuseTalk": MuseTalk,
    "LoadVideo": LoadVideo,
    "PreViewVideo": PreViewVideo,
    "CombineAudioVideo": CombineAudioVideo,
    "MuseTalkRealTime": MuseTalkRealTime
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MuseTalk": "MuseTalk Node",
    "LoadVideo": "Video Loader",
    "PreViewVideo": "PreView Video",
    "CombineAudioVideo": "Combine Audio Video",
    "MuseTalkRealTime": "MuseTalk RealTime Node"
}
