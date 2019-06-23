import cv2
import string
import numpy as np
import math

#Constant that converts camera units to inches
cameraUnitsToInches = 0.66317864799


#Declares Global variables
distanceLeft = 0
distanceRight = 0

rightAndLeft = 0

rightAngle =0
leftAngle=0

shortAct = 2
medAct = 5.5
longAct = math.sqrt(shortAct * shortAct + medAct * medAct)

thetaLeftAngle = 0
thetaRightAngle = 0

#Finds distance for a point in three dimensional space
def findDistance(p1, p2):
    dist = math.sqrt(math.pow((p1[0] - p2[0]),2) + math.pow((p1[1] - p2[1]),2) + math.pow((p1[2] - p2[2]),2))
    return dist

#Finds distance for a point in two dimensional space
def findDistance2d(p1, p2):
    dist = math.sqrt(math.pow((p1[0] - p2[0]),2) + math.pow((p1[1] - p2[1]),2))
    return dist

#Finds the order of three points
def findOrder(v1,v2,v3):
    if(v1 > v2 and v2 > v3):
        return v1, v2, v3

    if(v1 > v3 and v3 > v2):
        return v1,v3,v2

    if(v2 > v1 and v1 > v3):
        return v2, v1, v3

    if(v2 > v3 and v3 > v1):
        return v2, v3, v1

    if(v3 > v1 and v1 > v2):
        return v3, v1, v2

    return v3,v2,v1

#Finds the perimeter of a rectangle
def findPerimeter(box):
    d1 = findDistance2d(box[0],box[1])
    d2 = findDistance2d(box[0],box[2])
    d3 = findDistance2d(box[1],box[3])
    d4 = findDistance2d(box[2],box[3])
    return d1+d2+d3+d4

#Finds the area of a rectangle
def findArea(box):
    d1 = findDistance2d(box[0],box[1])
    d2 = findDistance2d(box[0],box[2])
    return d1*d2

#Calculates the error for gradient descent
def error(corner, long, short):
    global shortAct
    global medAct 
    global longAct 
    
    component1 = findDistance(long, short)/findDistance(long, corner) - longAct/medAct
    component2 = findDistance(long,short)/findDistance(short,corner) - longAct/shortAct
    component3 = findDistance(short, corner)/findDistance(long,corner) - shortAct/medAct
    return math.fabs(component1) + math.fabs(component2) + math.fabs(component3)

#Checks if the contour is a target
def isATarget(box, box2):
    minPerimeter = 50
    maxPerimeter = 800
    minArea = 600
    maxArea = 38000

    #box is left target
    #box2 is right target

    #If the perimeter of both rectangles is within the bounds of perimeter
    if(findPerimeter(box) > minPerimeter and findPerimeter(box) < maxPerimeter and findPerimeter(box2) > minPerimeter and findPerimeter(box2) < maxPerimeter):
        
        #If the area of both rectangles is within the bounds of area
        if(findArea(box) > minArea and findArea(box) < maxArea and findArea(box2) > minArea and findArea(box2) < maxArea ):
            
            #If the perimeter of each rectangle is not more than twice as large as the other
            if(findPerimeter(box) < 2*findPerimeter(box2) and findPerimeter(box) < 2*findPerimeter(box2)):
                    
                    #If the distance between points on the rectangle fits the orientation of a target
                    if(findDistance2d(box[0], box[1]) < findDistance2d(box[0], box[2]) and findDistance2d(box2[0], box2[1]) < findDistance2d(box2[0], box2[2])):
                        
                        #If the distance of a rectangle fits the orientation of a target
                        if( findDistance2d(box[1], box2[0]) > findDistance2d(box[3], box2[2])):

                            #If the targets are within 3 inches of each other
                            if(-3 < gradientDescent(box) - gradientDescent(box2) < 3):
                                return True
        
    return False



#The gradient descent algorithm to calculate distance and angle
def gradientDescent(coord):
    #Gets the points of the target
    a = np.array([coord[0][0], coord[0][1], 0])
    b = np.array([coord[1][0], coord[1][1], 0])
    c = np.array([coord[2][0], coord[2][1], 0])
    
    #Finds the distance between points and sets them to side lengths
    ab = findDistance(a,b)
    bc = findDistance(b,c)
    ac = findDistance(a,c)
    
    #Order the side lengths
    max, med, min = findOrder(ab,bc,ac)
    
    #Uses side length to figure out orientation of points 
    if (ac == max and ab == med):
        long = a
        short = c
        corner = b
    elif (ab == max and ac == med):
        long = a
        short = b
        corner = c
    elif(bc == max and ab == med):
        long = b
        short = c
        corner = a
    elif(ab == max and bc == med):
        long = b
        short = a
        corner = c
    elif(bc == max and ac == med):
        long = c
        short = b
        corner = a
    else:
        long = c
        short = a
        corner = b
    
    #Constants and Variables for gradient descent calculation
    z1 = 0
    z2 = 0
    deltaZ1 = .0001
    deltaZ2 = .0001
    
    
    #Uses gradient descent to rotate the points around in 3d space until the ratios of distance match up with measurements of the target
    for iteration in range(200):
        corner = np.array([corner[0], corner[1],z1])
        cornerDelta = np.array([corner[0], corner[1],z1 + deltaZ1])
    
        short = np.array([short[0], short[1], z2])
        shortDelta = np.array([short[0], short[1], z2 + deltaZ2])
    
        long = np.array([long[0], long[1], -z2])
        longDelta = np.array([long[0], long[1], -z2 - deltaZ2])
    
    
        partialZ1 = (error(cornerDelta, long, short) - error(corner, long, short))/deltaZ1
        partialZ2 = (error(corner, longDelta, shortDelta) - error(corner, long, short))/deltaZ2
        
        gradient = np.array([partialZ1, partialZ2])
        
        z1 -= partialZ1/np.linalg.norm(gradient) * .04
        z2 -= partialZ2/np.linalg.norm(gradient) * .04
    
        #If the error is small stop running gradient descent
        if(error(corner, long, short) < .01):
           break
    
        nothing = 0
    
    
    #Distances between points
    newLong = findDistance(short, long)
    newMed = findDistance(corner, long)
    newShort = findDistance(short, corner)
    
    #Call global variables
    global shortAct
    global medAct 
    global longAct 
    
    global cameraUnitsToInchesToInches
    
    global rightAndLeft
    
    #Calculates vectors for use in the crossproduct
    cornerToShort = short - corner
    cornerToLong = long - corner
    
    crossProduct = np.array([cornerToShort[1]*cornerToLong[2] - cornerToShort[2]*cornerToLong[1],
                             cornerToShort[2]*cornerToLong[0] - cornerToShort[0]*cornerToLong[2],
                             cornerToShort[0]*cornerToLong[1] - cornerToShort[1]*cornerToLong[0]])
    
    
    x = (coord[0][0] + coord[3][0])/2
    

    

    if(rightAndLeft == 0):
        #Calculates distance, converts it to inches, and stores it to a global var
        global distanceRight
        distanceRight = longAct/newLong * cameraUnitsToInches * 1000
        
        #Caclulates angle, converts to degress, and stores it to a global var
        global rightAngle
        rightAngle = (math.atan(crossProduct[0]/crossProduct[2])) * 180/math.pi

        #Caclulates the theta angle, converts to degress, and stores it to a global var
        global thetaRightAngle
        thetaRightAngle =  math.atan((x - 320)/453.461) * 180/math.pi
        
    else:
        #Calculates distance, converts it to inches, and stores it to a global var
        global distanceLeft
        distanceLeft = longAct/newLong * cameraUnitsToInches * 1000
        
        #Caclulates angle, converts to degress, and stores it to a global var
        global leftAngle
        leftAngle = (math.atan(crossProduct[0]/crossProduct[2])) * 180/math.pi

        #Caclulates the theta angle, converts to degress, and stores it to a global var
        global thetaLeftAngle
        thetaLeftAngle = math.atan((x - 320)/453.461) * 180/math.pi

    rightAndLeft += 1
    
    #Returns the distance to the target
    return longAct/newLong * cameraUnitsToInches * 1000


coord = None


    #Cycles through all of the contours found and finds the pair that is a target
def contourFunction(contours):
    
    foundTargets = False

    for i in range(0,len(contours)):
        
            #Checks the solidity of the contour
            if(float(cv2.contourArea(contours[i]))/(cv2.contourArea(cv2.convexHull(contours[i])) + -.0001) > .80):
                
                rect = cv2.minAreaRect(contours[i])
                box = cv2.boxPoints(rrect)
                box = np.int0(bbox)
    
                
                for j in range(0,len(contours)):
                    if(i != j):

                        rect2 = cv2.minAreaRect(contours[j])
                        box2 = cv2.boxPoints(rrect2)
                        box2 = np.int0(bbox2)
                        
                        #Checks if the pair is a target
                        if(isATarget(box,box2)):   
                            print('left one' , box)
                            print('right one' , box2)
                            
                            gradientDescent(box)
                            gradientDescent(box2)

                            foundTargets = True
                            break

                if(foundTargets):
                    break



    x = 0

#Returns the distance and angle when an image or videostream is inserted
def getVals(cap):
    count = 0;

    #Upper and lower values for HSV filter
    lowerVal = 240
    upperVal = 255

    key = cv2.waitKey(1)

    #Creates the frame
    haveFrame, frame = cap.read()

    #Converts to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Masks the reflective tape
    mask = cv2.inRange(hsv, (0, 0, lowerVal) , (180, 160, upperVal))
    
    #Finds contours
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]


    #Calls the contour function to find targets
    contourFunction(contours)


    #Gets the global variables
    global distanceLeft, distanceRight, rightAngle, leftAngle, thetaLeftAngle, thetaRightAngle\

    #Calculates distance from the center of the target
    distance = (distanceLeft + distanceRight) / 2
    print('\n')
    print('distance in inches = ', distance)

    #Calculates the angle from the center of the target
    angle = (rightAngle + leftAngle) / 2
    print('\n')
    print('angle = ', angle)

    #Calculates the theta angle from the center of the target
    thetaAngle = (thetaRightAngle + thetaLeftAngle) / 2
    print('\n')
    print('thetaangle = ', thetaAngle)

    #Returns the distance and angles
    return distance, angle, thetaAngle
