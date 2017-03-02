# 获取携程的验证码
import urllib.request
begin = 1488163750792
img_url = 'http://m.ctrip.com/restapi/searchapi/appcaptcha/h5/captcha?v='
path = 'e:/img/'
count = 1
for i in range(0,10000):
    try:
        img_data = urllib.request.urlopen(img_url+str(begin+count)).read()
        f = open(path+str(count)+'.png','wb+')
        f.write(img_data)
        f.close()
        count +=1
    except urllib.request.URLError as e:
        print(e.reason)
