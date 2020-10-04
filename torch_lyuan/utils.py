import random

def collate_fn_(batch_data, max_len=40000):
    audio = batch_data[0]
    audio_len = audio.size(1)
    if audio_len > max_len:
        idx = random.randint(0,audio_len - max_len)
        return audio[:,idx:idx+max_len]
    else:
        return audio