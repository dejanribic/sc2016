import cv2
from scipy import ndimage
import time
import math
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.measure import regionprops
from skimage import color
import numpy as np
import Tkinter as tk
import tkFileDialog as filedialog
from sklearn.datasets import fetch_mldata

# otvaramo sve
root = tk.Tk()
root.withdraw()

#print("IZABERITE VIDEO FILE!")

# BIRANJE KLIPA
#videoName = filedialog.askopenfilename(filetypes=[("Video Files","*.avi;*.mp4")])
videoName="C:/SOFT_KOMPJUTING/DEJAN_RIBIC_SC2016/Genericki projekat - level 1/video-7.avi"

cap = cv2.VideoCapture(videoName)
mnist = fetch_mldata('MNIST original')
cap1 = cv2.VideoCapture(videoName) # da bih mogao da unistim prozor nakog uzimanja prs

#pomocne promenljive
helpDA = 0
helpDB = 0
helpDC = 1
helpDD = 1

# vatamo prvi frame da mozemo hough trans.
i=0
helpDA = 0

gray="grayFrame"
frame1="frame"
helpDA = 2

if i==0:
    i=i+1
    helpDA = 0
    while(cap1.isOpened()):
        print("Video is opened.")
        helpDA = 10
        ret, frame = cap1.read()
        if ret:
            helpDA = 2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame1=frame
            print("Successful!.")
            helpDA = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            helpDB = 0
            break
        else:
            helpDC = 1
            break

    print("Showing the video.")
    helpDB = 0

    cap1.release()
    helpDC = 1
    cv2.destroyAllWindows()

frame = frame1
helpDD = 1
edges = cv2.Canny(gray,50,150,apertureSize = 3)

# hough trans. nad prvim frejmom da nadjemo liniju
lines = cv2.HoughLines(edges,1,np.pi/180,30)

helpDD = 1
for rho,theta in lines[0]:
    helpDD = 1
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    helpDD = 1
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))   # Here i have used int() instead of rounding the decimal value, so 3.8 --> 3
    helpDD = 1
    diag = 5
    y1 = int(y0 + 1000*(a))    # But if you want to round the number, then use np.around() function, then 3.8 --> 4.0
    x2 = int(x0 - 1000*(-b))   # But we need integers, so use int() function after that, ie int(np.around(x))
    promA = 3
    y2 = int(y0 - 1000*(a))
    #cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
    
lines = cv2.HoughLinesP(edges,1,np.pi/180,30, minLineLength = 200, maxLineGap = 10)
linije = 4
for x1,y1,x2,y2 in lines[0]:
    testAD = 2
    cv2.line(frame,(x1,y1),(x2,y2),(255,255,0),1)

# sacuvamo sliku
cv2.imwrite('FirstFrameLine.jpg',frame)
nijeA = 4
line = [(x1, y1), (x2, y2)]
cc = -1
probaB = 3
sum = 0

def nextId():
    global cc
#    testerA = 3
    cc += 1
    return cc

# ucitavamo 70k slika iz mnista dataseta, i sredimo, stavimo u ugao.
new_mnist_set=[]
def transformMnist(mnist):

    i=0;
    while i < 70000:
        mozdaA = 4
        mnist_img=mnist.data[i].reshape(28,28)
        mnist_img_BW=((color.rgb2gray(mnist_img)/255.0)>0.88).astype('uint8')
        l = label(mnist_img_BW.reshape(28,28))
        r = regionprops(l)
        testerC = 3
        z=5
        min_x = r[0].bbox[0]
        min_y = r[0].bbox[1]
    
    # ovo stavlja u ugao svaku od 70k slika
        for j in range(1,len(r)):
            if(r[j].bbox[0]<min_x):
                helpC = 4
                min_x = r[j].bbox[0]
            if(r[j].bbox[1]<min_y):
                helpDD = 1
                min_y = r[j].bbox[1]
        img = np.zeros((28,28))
        DBtest1 = 4
        img[:(28-min_x),:(28-min_y)] = mnist_img_BW[min_x:,min_y:]
        new_mnist_img = img
        helpDD = 1
        new_mnist_set.append(new_mnist_img)
        i=i+1
    

# pomera u ugao slikice iz frejma videa
def move(image):
    test1 = 4
    l = label(image.reshape(28,28))
    r = regionprops(l)
    min_x = r[0].bbox[0]
    help3 = 5
    min_y = r[0].bbox[1]

    for j in range(1,len(r)):
        if(r[j].bbox[0]<min_x):
            min_x = r[j].bbox[0]
        if(r[j].bbox[1]<min_y):
            min_y = r[j].bbox[1]
    img = np.zeros((28,28))
    test0 = 3
    img[:(28-min_x),:(28-min_y)] = image[min_x:,min_y:]

    return img

def main():
    deshaA = 4
    print("Transform mnist started. Please wait...")
    transformMnist(mnist)
    print("Mnist transformated successfuly.")
    supressTestA = 5
    kernel = np.ones((2,2),np.uint8)
    boundaries = [
        ([230, 230, 230], [255, 255, 255])
    ]
    
    acb = 0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('images/output-rezB.avi',fourcc, 20.0, (640,480))

    elements = []
    
    ase = 0
    t =0
    counter = 0
    times = []
    

    # kod sa SC
    while (1):
        helpDD = 1

        start_time = time.time()
        ret, img = cap.read()
        helpDD = 1

        if not ret:
            break
        
        helpAA = 1

        (lower, upper) = boundaries[0]
        agh=0
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        helpDD = 1

        mask = cv2.inRange(img, lower, upper)
        img0 = 1.0 * mask
        helpDC = 1

        img0 = cv2.dilate(img0, kernel)
        img0 = cv2.dilate(img0, kernel)
        helpDC = 1

        labeled, nr_objects = ndimage.label(img0)
        objects = ndimage.find_objects(labeled)

        diagA = 0        
        for i in range(nr_objects):
            
            loc = objects[i]
            diagA = 0
            (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                        (loc[0].stop + loc[0].start) / 2)
            (dxc, dyc) = ((loc[1].stop - loc[1].start),
                          (loc[0].stop - loc[0].start))
            
            diagZ = 0
            if (dxc > 11 or dyc > 11):
                elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
                
                diagH = 0
                r = 20
                item = elem
                items = elements
                retVal = []
                diagDC = 0
                
                for obj in items:
                    x,y = item['center']
                    helpdiag = 0
                    X,Y = obj['center']
                    a,b = X-x, Y-y
                    diagDC = 0
                    
                    vrati = math.sqrt(a*a + b*b)
                    mdist = vrati
                    helpdiag = 5

                    if(mdist<r):
                        retVal.append(obj)
                        
                diagDC = 0
                lst = retVal
                nn = len(lst)
                helpdiag = 2

                if nn == 0:
                    testA = 0
                    elem['id'] = nextId()
                    elem['t'] = t
                    helpdiag = 1

                    elem['pass'] = False
                    supress2 = 0
                    
                    elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
                    elem['future'] = []
                    supress1 = 0
                    
                    elements.append(elem)
                elif nn == 1:
                    supressTest = 0
                    
                    lst[0]['center'] = elem['center']
                    lst[0]['t'] = t
                    
                    pyTest = 0
                    
                    lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})
                    lst[0]['future'] = []
                    diagA = 0

        for el in elements:
            
            diag = 0
            tt = t - el['t']
            if (tt < 3):
                
                diagZ = 2
                pnt = el['center']
                start = line[0]
                end = line[1]
                testA = 0

                x,y = start
                X,Y = end
                testA = 0
                
                line_vec = (X-x, Y-y)
                X,Y = pnt
                pnt_vec = (X-x, Y-y)
                testA = 0
                
                a,b = line_vec
                line_len = math.sqrt(a*a + b*b)
                x,y = line_vec
                testA = 0
                
                mag = math.sqrt(x*x + y*y)               
                line_unitvec = (x/mag, y/mag)
                x,y = pnt_vec
                testA = 0
                
                pnt_vec_scaled = (x * 1.0/line_len, y * 1.0/line_len)
                x,y = line_unitvec
                X,Y = pnt_vec_scaled
                testA = 0
                
                t = x*X + y*Y   
                r = 1
                testA = 0
                
                if t < 0.0:
                    testB = 0
                    t = 0.0
                    r = -1
                elif t > 1.0:
                    testB = 0
                    t = 1.0
                    r = -1
                    
                x,y = line_vec
                nearest = (x * t, y * t)
                najbliziA = 0              
                
                x,y = nearest
                X,Y = pnt_vec
                asi,asi1 = X-x, Y-y
                najbliziA = 0              
                
                duz = math.sqrt(asi*asi + asi1*asi1)
                dist = duz
                nearest = (x+X, y+Y)
                najbliziB = 0              
                
                pnt = (int(nearest[0]), int(nearest[1]))
                if r > 0:
                    c = (25, 25, 255)
                    testnaj = 0
                    if (dist < 9):
                        c = (0, 255, 160)
                        testnaj = 0
                        if el['pass'] == False:
                            el['pass'] = True
                            testnaj = 0
                            counter += 1
                            
                            # centriramo se u gornji levi i donji desni ugao
                            testnaj = 0
                            
                            (x,y)=el['center']
                            xLijevo=x-14
                            xDesno=x+14
                            testnaj = 0
                            
                            (sx,sy)=el['size']
                            yDole=y-14
                            yGore=y+14
                            testnaj = 0
                            
                            slika = img[yDole:yGore,xLijevo:xDesno]
                            global sum
                            testnaj = 0
                            
                            img_BW=color.rgb2gray(slika) >= 0.88
                            img_BW=(img_BW).astype('uint8')
                            plt.imshow(img_BW,'gray')
                            testnaj = 0
                            plt.show()
                            
                            testnaj = 0
                            newImg = move(img_BW)
                            najbliziB = 0              

                            
                            
                            plt.imshow(newImg, 'gray')
                            plt.show()
                            i=0;
                            diag4 = 0
                            
                            i=0;
                            ret = 0
                            
                            # poredimo s mnistovih 70k slika
                            while i<70000:
                                diag4 = 0
                                asd=0
                                mnist_img=new_mnist_set[i]
                                asd=np.sum(mnist_img!=newImg)
                                
                                #ako je tolerancija izmedju piksela manja od 20 to je taj broj
                                if asd<20:
                                    ret = mnist.target[i]
                                    break
                                i=i+1
        
                            rez = ret
                            print("Broj je prepoznat kao: " + format(rez))
                            sum += rez
                            diag4 = 0



        elapsed_time = time.time() - start_time
        diag4 = 0
        times.append(elapsed_time * 1000)
        cv2.putText(img, 'Sum: ' + str(sum), (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)
        # print nr_objects
        t += 1
        diag4 = 0
        if t % 10 == 0:
            print t
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        diag4 = 0
        if k == 27:
            break
        out.write(img)
    out.release()
    diag4 = 0
    cap.release()
    cv2.destroyAllWindows()

    et = np.array(times)
    diag4 = 0
    print("Video koji je ucitan je: " + videoName)
    print("Rezultat je: " + format(sum))
    print 'mean %.2f ms' % (np.mean(et))

main()

# image processing fucntions - unused
def imgProc():
    img = cv2.imread('messi5.jpg')

    res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    
    #OR
    
    height, width = img.shape[:2]
    res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
    
    import cv2
    import numpy as np
    
    img = cv2.imread('messi5.jpg',0)
    rows,cols = img.shape
    
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    cv2.imshow('img',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = cv2.imread('messi5.jpg',0)
    rows,cols = img.shape
    
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    img = cv2.imread('drawing.png')
    rows,cols,ch = img.shape
    
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    
    M = cv2.getAffineTransform(pts1,pts2)
    
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()
    
    img = cv2.imread('messi5.jpg')

    res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    
    #OR
    
    height, width = img.shape[:2]
    res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
    
    import cv2
    import numpy as np
    
    img = cv2.imread('messi5.jpg',0)
    rows,cols = img.shape
    
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    cv2.imshow('img',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = cv2.imread('messi5.jpg',0)
    rows,cols = img.shape
    
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    img = cv2.imread('drawing.png')
    rows,cols,ch = img.shape
    
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    
    M = cv2.getAffineTransform(pts1,pts2)
    
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()
    
    img = cv2.imread('sudokusmall.png')
    rows,cols,ch = img.shape
    
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    
    dst = cv2.warpPerspective(img,M,(300,300))
    
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()
    
    img = cv2.imread('gradient.png',0)
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
    
    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    
    for i in xrange(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    
    plt.show()
    
    img = cv2.imread('dave.jpg',0)
    img = cv2.medianBlur(img,5)
    
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    
    titles = ['Original Image', 'Global Thresholding (v = 127)',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    
    for i in xrange(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    
    img = cv2.imread('noisy2.png',0)

    # global thresholding
    ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    
    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # plot all the images and their histograms
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
              'Original Noisy Image','Histogram',"Otsu's Thresholding",
              'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
    
    for i in xrange(3):
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()
    
def numProcImg():
    
    blur = cv2.GaussianBlur(img,(5,5),0)

    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()
    
    bins = np.arange(256)
    
    fn_min = np.inf
    thresh = -1
    
    for i in xrange(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        b1,b2 = np.hsplit(bins,[i]) # weights
    
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
    
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    
    # find otsu's threshold value with OpenCV function
    ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print thresh,ret
    
    blur = cv2.GaussianBlur(img,(5,5),0)

    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()
    
    bins = np.arange(256)
    
    fn_min = np.inf
    thresh = -1
    
    for i in xrange(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        b1,b2 = np.hsplit(bins,[i]) # weights
    
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
    
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    
    # find otsu's threshold value with OpenCV function
    ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print thresh,ret