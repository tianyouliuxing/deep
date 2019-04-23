from django.shortcuts import render
from django.shortcuts import HttpResponse
# Create your views here.
import tensorflow as tf
import json
import os

user_list = [
    {"user":"jack","pwd":"abc"},{"user":"tom","pwd":"ABC"},
]


def index(request):
    #request.POST
    #request.GET
    #return HttpResponse("hello world")
        if request.method=="POST":
            username =request.POST.get("keyword",None)
           # password =request.POST.get("password",None)
          #  print(username,password,"hahahah")

            print(username, "hahahah")
           # temp ={"user":username,"pwd":password}
            #user_list.append(temp)
        #return render(request,"index.html",{"data":user_list})
        return render(request,"index.html")

def cat(request):
    #request.POST
    #request.GET
    #return HttpResponse("hello world")
        print("45", "hahahah")
        if request.method=="POST":
            ret = {'status': False, 'data': None, 'error': None}
            try:
                user = request.POST.get('user')
                img = request.FILES.get('img')
                f = open(os.path.join('static', img.name), 'wb')
                for chunk in img.chunks(chunk_size=1024):
                    f.write(chunk)
                ret['status'] = True
                ret['data'] = os.path.join('static', img.name)
            except Exception as e:
                ret['error'] = e
            finally:
                f.close()
                return HttpResponse(json.dumps(ret))
        return render(request, 'catAndDog.html')
        #return render(request,"catAndDog.html",{"data":user_list})

def index1(request):
    #request.POST
    #request.GET
    #return HttpResponse("hello world")
        if request.method=="POST":
            username =request.POST.get("keyword",None)
           # password =request.POST.get("password",None)
          #  print(username,password,"hahahah")
            print(username, "hahahah")
           # temp ={"user":username,"pwd":password}
            #user_list.append(temp)
        #return render(request,"index.html",{"data":user_list})
        return render(request,"show.html")

def editProfile(request):
    if request.method == 'GET':
        return render(request, 'upload.html')
    if request.method == 'POST':
        file = request.FILES.get('file')  # 获取文件信息用 request.FILES.get
        print(file)  # 这里的get('file') 相当于 name = file
        # print(file) 可以直接显示文件名，是因为django FILES内部 重写了 __repr__ 方法
        if file:  # 如果文件存在
            try:
                with open("D:\\Pycharm\\untitled7\\picture\\" + file.name, 'wb') as f:  # 新建1张图片 ，图片名称为 上传的文件名
                    for temp in file.chunks():  # 往图片添加图片信息
                        f.write(temp)
            except EOFError:
                print("File is not found.")
            except PermissionError:
                print("no permission")
        import sys
        sys.path.append("first_app/modelTrain")
        import face_test
        a =face_test.predict(file.name)
        print(a,"result")
            #from modelTrain import hah
            #os.system("modelTrain/hah.py")
        return HttpResponse(a)

def allow_file(filename):
    allow_list = ['png', 'PNG', 'jpg', 'doc', 'docx', 'txt', 'pdf', 'PDF', 'xls', 'rar', 'exe', 'md', 'zip']
    a = filename.split('.')[1]
    if a in allow_list:
        return True
    else:
        return False