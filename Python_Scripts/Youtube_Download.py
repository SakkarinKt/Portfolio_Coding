#pip install pytube

from pytube import YouTube

def Download(url):
    youtubeObject = YouTube(url)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download()
    except:
        print("Something wrong")
    print("Download success!!")

link = input("Enter the Youtube video link: ")
Download(link)