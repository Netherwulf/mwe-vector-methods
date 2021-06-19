import requests


def download(url):
    get_response = requests.get(url, stream=True)

    file_name = url.split("=")[-1]

    totalbits = 0

    with open(file_name, 'wb') as f:
        for chunk in get_response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                totalbits += 1024
                print("Downloaded", totalbits * 1025, "KB...")
                f.write(chunk)


# download("https://example.com/example.jpg")

if __name__ == "__main__":
    url = 'https://nextcloud.clarin-pl.eu/index.php/s/luubhnS0AvjmtQc/download?path=%2F&files=kgr10.plain.skipgram.dim300.neg10.bin'

    download(url)
