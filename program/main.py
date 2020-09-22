import cv2
from opencv_visualize import display_instances, class_names
from model_generator import model

# import pandas for dataset manipulation
import pandas as pd
from datetime import datetime

# import the necessary packages for zmq
import imagezmq



# initialize the ImageHub object
imageHub = imagezmq.ImageHub(open_port='tcp://*:5555')

def howManyPeople(list):
    count=0
    for object in list:
        if (object==1): #1 is the ID for 'person'
            count=count+1
    return count

def returnTime():
    # current date and time
    now = datetime.now().isoformat(' ', 'seconds')
    return now

def writeInCSV(classList):
    people = howManyPeople(classList)
    if(people>=1):
        dataset = pd.read_csv("contadorPersonas.csv") #read csv
        df2 = pd.DataFrame([[returnTime(), people]], columns=['event_time','people']) #create dataframe with time and people
        dataset = pd.concat([dataset, df2], sort=False, ignore_index=True) # concat csv with the newone
        dataset.to_csv('contadorPersonas.csv',index_label=False,index=False) # write the new info into the csv

if __name__ == '__main__':
    while True:
        (clientName, frameReceived) = imageHub.recv_image()
        imageHub.send_reply(b'OK')
        print(clientName)
        cv2.imwrite('frameReceived.jpeg',frameReceived)
        frame = cv2.imread('frameReceived.jpeg', cv2.IMREAD_COLOR)

        results = model.detect([frame], verbose=0)
        r = results[0]

        writeInCSV(r['class_ids'])

        #display the image with the mask
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

