from processdata import sent_embedding
from scrapper import get_text_content
from time import sleep

print('module loaded')

print( sent_embedding(get_text_content('google.com')['content']))

sleep(5)