#run hatch area detection
im = cv2.imread(sys.argv[1])
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
white = 255
black = 0
#im = cv2.copyMakeBorder(im,10,10,10,10,cv2.BORDER_CONSTANT, value = white)
im_copy = im.copy()
cv2.imshow('original', im_copy)
im = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)

kernel_1 = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]])

ob = preprocess(im)
im = ob.GaussianBlur()
#im = cv2.filter2D(im, -1, kernel)
ob = preprocess(im)
im = ob.Blur(5)
im = cv2.medianBlur(im, 7)
ob = preprocess(im)
im = ob.Blur(5)

im = cv2.bilateralFilter(im, 9, 10, 10)
im = cv2.bilateralFilter(im, 9, 75, 75)
im = cv2.bitwise_not(im)
im = cv2.erode(im, kernel_1, iterations = 11)
im = cv2.dilate(im, kernel_1, iterations = 2)
im = cv2.medianBlur(im, 5)

#mean = np.mean(im)
#print(mean)
mask = np.zeros_like(im)
mask [im>10] = 255
#mask = cv2.bitwise_and(mask, tex)

'''
kernel_2 = np.ones((7,7))
mask = cv2.dilate(im, kernel_2, iterations = 2)
im_not = cv2.bitwise_not(mask)
cv2.imshow("base",cv2.bitwise_and(im_not, im_copy))

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
i=0
for c in contours:
    original_image_copy = np.zeros_like(im)
    copy_img = im_copy.copy()
    cv2.drawContours(original_image_copy, contours, i, 255, -1)
    show_img = cv2.bitwise_and(original_image_copy, copy_img)
    cv2.imshow("Region "+str(i), show_img)
    del copy_img
    del original_image_copy
    i+=1
'''
cv2.imshow('see', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
