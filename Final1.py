import numpy as np
import cv2
import skimage.morphology
import skimage.measure


help_message = '''
Keys:
 1 - toggle Local Binary Pattern included
 2 - toggle showing Local Binary Pattern
 q - to quit the program
'''


cam = cv2.VideoCapture(0)

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#Create the grid of points
def createGrid(frame,step = 20):
    grid = []
    h, w = frame.shape[:2]
    y,x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    for (y1,x1) in zip(y,x):
        grid.append([[np.float32(x1),np.float32(y1)]])
    return np.array(grid)

old_frame = cam.read()[1]
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_gray = cv2.GaussianBlur(old_gray, (21, 21), 0)

p_old = createGrid(old_frame)
grid = np.uint32(p_old.reshape(768,2))
coords = grid.reshape(24,32,2).astype(np.uint32)


#Filter out unreliable optical flow vectors.
def filter(p_new,st):
    d = abs(p_new - p_old).reshape(-1, 2).max(-1)
    for i, d1 in enumerate(d):
        if d1 > 70:
            p_new[i] = p_old[i]
            st[i] = 0
        elif d1 < 6:
            p_new[i] = p_old[i]
            st[i] = 0


#Visualize optical flow vector
def draw(p_new_valid,p_old_valid,frame):
    lines = np.hstack([p_old_valid,p_new_valid]).reshape(-1,2,2)
    lines = np.int32(lines)

    cv2.polylines(frame, lines, 0, (0, 255, 0))
    for (x1, y1) in grid:
        cv2.circle(frame, (x1, y1), 2, (0, 255, 0), -1)
    return frame


#Group pixels in motion together to find the moving object. No LBP is included.
def grouping(st,frame):
    group = []
    st = st.reshape(24,32)
    labeled = skimage.morphology.label(st)
    labeled = skimage.morphology.remove_small_objects(labeled, 5,connectivity=2)

    props = skimage.measure.regionprops(labeled)
    for prop in props:
        pos = prop.bbox

        cv2.rectangle(frame,(tuple(coords[pos[0],pos[1]])),(tuple(coords[pos[2]-1,pos[3]-1])),(0,255,0),2)
        group.append(pos)

    group = np.array(group).reshape(-1,4)
    return group

#Check overlap between motion field and similar-texture field
def checkOverlap(l1_x,l1_y,r1_x,r1_y,l2_x,l2_y,r2_x,r2_y):
    if (l1_x > r2_x) or (l2_x > r1_x):
        return False
    if (l1_y > r2_y) or (l2_y > r1_y):
        return False
    return True

#Combine LBP and optical flow to detect moving object
def combination(texture,motion,frame):
    labeled = skimage.morphology.label(texture)
    labeled = skimage.morphology.remove_small_objects(labeled, 25,connectivity=2)
    out =   labeled.copy()
    component_sizes = np.bincount(labeled.ravel())
    too_big = component_sizes > 45
    too_big_mask = too_big[labeled]
    out[too_big_mask] = 0
    props = skimage.measure.regionprops(out)
    for prop in props:
        pos = prop.bbox
        cv2.rectangle(frame, (tuple(coords[pos[0], pos[1]])), (tuple(coords[pos[2] - 1, pos[3] - 1])), (0, 0, 255), 2)
        overlap=0
        for region in motion:
            if (checkOverlap(pos[1],pos[0],pos[3]-1,pos[2]-1,region[0],region[1],region[2]-1,region[3]-1)):
                if overlap ==0:
                    top_x = min(pos[0], region[0])
                    top_y = min(pos[1], region[1])
                    bottom_x = max((pos[2] - 1), (region[2] - 1))
                    bottom_y = max((pos[3] - 1), (region[3] - 1))
                else:
                    top_x = min(top_x,region[0])
                    top_y = min(top_y,region[1])
                    bottom_x = max(bottom_x, (region[2] - 1))
                    bottom_y = max(bottom_y, (region[3] - 1))

                overlap = overlap + 1

                if overlap ==2:
                    cv2.rectangle(frame, tuple(coords[top_x,top_y]), tuple(coords[bottom_x,bottom_y]),(255, 0, 0), 2)

#Function to find the local binary pattern of a pixel
def LBP(x,y,image):
    roi = image[y-1:y+2,x-1:x+2]
    roi = roi.flatten()
    center = roi[4]
    roi = np.delete(roi,4)
    for i,element in enumerate(roi):
        if roi[i]>center:
            roi[i] = 0
        else:
            roi[i] = 1
    decimal = roi.dot(2 ** np.arange(roi.size)[::-1])
    return decimal


#Function to find LBP of all pixels in the grid
def Texture(frame):
    lbp=np.zeros((768,1),dtype=np.uint8)
    for i,coords in enumerate(grid):
        lbp[i] = LBP(coords[0],coords[1],frame)
    lbp = lbp.reshape(24,32,1)

    return lbp



def main(old_gray,p_old):
    print help_message

    show_lbp = True
    lbp_included = True

    while True:
        (grabbed, frame) = cam.read()
        if not grabbed:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
        p_new, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p_old, None, **lk_params)

        filter(p_new,st)
        test = frame.copy()
        group = grouping(st,test)

        lbp = Texture(frame_gray)

        if lbp_included:
            combination(lbp,group,test)

        if show_lbp:
            lbp = cv2.resize(lbp, (0, 0), fx=5, fy=5)
            cv2.imshow("LBP",lbp)

        cv2.imshow("Optical flow",draw(p_new[st==1],p_old[st==1],frame))


        cv2.imshow("Test",test)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord('1'):
            lbp_included = not lbp_included
            print 'LBP is', ['not included', 'incuded'][lbp_included]
        if key == ord('2'):
            show_lbp = not show_lbp
            print 'LBP is', ['off', 'on'][show_lbp]
        old_gray = frame_gray.copy()

    cv2.destroyAllWindows()
    cam.release()

if __name__ == "__main__":
    main(old_gray,p_old)