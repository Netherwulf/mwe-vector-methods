import urllib.request

if __name__ == "__main__":
    url = 'https://nextcloud.clarin-pl.eu/index.php/s/luubhnS0AvjmtQc/download?path=%2F&files=kgr10.plain.skipgram.dim300.neg10.bin'
    response = urllib.request.urlopen(url)
    html = response.read()