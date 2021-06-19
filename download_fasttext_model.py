import urllib.request

if __name__ == "__main__":
    url = 'https://nextcloud.clarin-pl.eu/index.php/s/luubhnS0AvjmtQc/download?path=%2F&files=kgr10.plain.skipgram.dim300.neg10.bin'
    response = urllib.request.urlopen(url)

    local_filename = url.split('=')[-1]

    totalbits = 0

    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    totalbits += 1024
                    print("Downloaded", totalbits * 1025, "KB...")
                    f.write(chunk)

    # html = response.read()
