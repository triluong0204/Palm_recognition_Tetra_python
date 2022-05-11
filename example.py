# -*- coding: UTF-8 -*-

from PROIE import *
import imutils

if __name__ == '__main__':
    # #####
    # #path_in_img = "resources/palmprint.jpg"
    

    # path_in_img = "test/a.jpg"


    # in_img_c = cv2.imread(path_in_img)

    # in_img_c = cv2.cvtColor(in_img_c, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("farm", in_img_c)
    # cv2.waitKey(0)

    # proie = PROIE()

    # proie.extract_roi(path_in_img, rotate=True)
    # proie.show_result()
    # proie.save("hihi/palmprint_roi.jpg")

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        try:
            proie = PROIE()

            roi_img = proie.extract_roi(frame, rotate=True)
            #proie.show_result()

            # Display the resulting frame
            cv2.imshow('frame',roi_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    proie.save("hihi/palmprint_roi.jpg")
    cap.release()
    cv2.destroyAllWindows()