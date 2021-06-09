import fasttext

def load_fasttext():
    model = fasttext.load_model("kgr10.plain.skipgram.dim300.neg10.bin")

    return model