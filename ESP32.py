import numpy as np
import cv2
import cv2.aruco as aruco
import socket
import heapq
import time

s = socket.socket()
host = '192.168.1.11'
port = 10000
try:
    s.connect((host, port))
except Exception:
    print('connection fail')
position = {}
orientation = {}


class Agent():
    def __init__(self, id, state=0, test=False) -> None:
        self.id = id
        self.state = state
        self.position = np.inf
        self.orientation = np.inf
        self.tick = 0
        if test:
            self.path = [15, 16]
        pass

    def set_location(self):
        if self.id in position:
            self.position = position[self.id]

    def set_orientation(self):
        if self.id in orientation:
            self.orientation = orientation[self.id]

    def set_path(self, path):
        self.path = path

    def forward(self):
        msg = str.encode('w')
        s.send(msg)
        print('Agent {}: forward...'.format(self.id))
        pass

    def backward(self):
        msg = str.encode('s')
        s.send(msg)
        print('Agent {}: backward...'.format(self.id))
        pass

    def turn_right(self):
        msg = str.encode('d')
        s.send(msg)
        print('Agent {}: right...'.format(self.id))
        pass

    def turn_left(self):
        msg = str.encode('a')
        s.send(msg)
        print('Agent {}: left...'.format(self.id))
        pass

    def stop(self):
        msg = str.encode('t')
        s.send(msg)
        print('Agent {}: stopping...'.format(self.id))
        pass

    def climb(self):
        pass

    def reach(self, target):
        if cal_distance(target, self.id, position) < 0.05:
            return True
        else:
            return False

    def head_to(self, id):
        v1 = position[id] - position[self.id]
        v2 = np.array([1, 0])
        cos_angle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(cos_angle) / np.pi * 180
        if v1[1] < 0:
            angle *= -1
        if self.orientation - angle < 4 and self.orientation - angle > -4:
            return True
        else:
            return False

    def set_state(self, new_state):
        self.state = new_state

    def state_control(self):
        '''
        state = 0, initialization
        state = 11, calculate the rotational angle and send the commend
        stage = 12, wait for the movement
        state = 21, calculate the forward distance and send the commend
        stage = 22, wait for the movement
        '''
        if self.state == 0:
            # initialization
            self.come_from = self.path.pop(0)
            self.target = self.path.pop(0)
            time.sleep(3)
            self.set_state(3)

        if self.state == 11:
            self.forward()
            self.set_state(12)

        if self.state == 12:
            if self.reach(self.target):
                self.set_state(3)
                self.come_from = self.target
                if self.path:
                    self.target = self.path.pop(0)
                else:
                    self.set_state(-1)
            else:
                # self.forward()
                if self.tick % 50 == 0:
                    if self.head_to(self.target):
                        self.set_state(12)
                    else:
                        self.set_state(21)
                else:
                    self.set_state(12)

        if self.state == 21:
            v1 = position[self.target] - position[self.id]
            v2 = np.array([1, 0])
            cos_angle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(cos_angle) / np.pi * 180
            if v1[1] < 0:
                angle *= -1
            agent_ori = self.orientation
            # print(angle)
            # print(agent_ori)
            if abs(angle - agent_ori) > 180:
                if angle > agent_ori:
                    self.turn_left()
                else:
                    self.turn_right()
            else:
                if angle < agent_ori:
                    self.turn_left()
                else:
                    self.turn_right()
            self.set_state(22)

        if self.state == 22:
            if self.head_to(self.target):
                self.set_state(3)
            else:
                # self.turn_right()
                self.set_state(22)

        if self.state == 3:
            self.stop()
            if self.head_to(self.target):
                self.set_state(11)
            else:
                self.set_state(21)

        if self.state == -1:
            self.stop()
            print('The agent arrives the target')

        self.tick += 1


class Graph():
    def __init__(self, nodes) -> None:
        self.graph = np.zeros((nodes, nodes))
        pass

    def add_edge(self, i, j, w):
        self.graph[i][j] = w
        self.graph[j][i] = w
        pass

    def find_path(self, s, e):
        pqueue = []
        heapq.heappush(pqueue, (0, s))
        seen = set()
        parent = {s: None}
        distance = [np.inf] * self.graph.shape[0]
        distance[s] = 0

        while len(pqueue) > 0:
            pair = heapq.heappop(pqueue)
            dist = pair[0]
            vertex = pair[1]
            seen.add(vertex)

            for i in range(self.graph.shape[0]):
                if self.graph[vertex][i] != 0:
                    if i not in seen:
                        if dist + self.graph[vertex][i] < distance[i]:
                            parent[i] = vertex
                            distance[i] = dist + self.graph[vertex][i]
                            heapq.heappush(pqueue, (distance[i], i))

        come_from = e
        self.path = []
        while come_from is not None:
            self.path.append(come_from)
            come_from = parent[come_from]
        self.path.reverse()


def open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)
    return cap


def init_parameters():
    mtx = np.array([[1051.1, 0, 695.0741],
                    [0, 1052.2, 297.7604],
                    [0., 0., 1.]])
    dist = np.array([[-0.4223, 0.1412, 0, 0, 0.0921]])
    return mtx, dist


def capture_frame(cap):
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame


def detect_aruco(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters_create()
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    corners, ids, rIP = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    return corners, ids, rIP


def get_position(ids, tvec, position):
    for i in range(ids.shape[0]):
        position[ids[i][0]] = (tvec[i][0])[:2]


def get_orientation(ids, rvec, orientation):
    for i in range(ids.shape[0]):
        temp = rvec[i][0]
        r, _ = cv2.Rodrigues(temp)
        theta_z = np.arctan2(r[1][0], r[0][0]) / np.pi * 180
        orientation[ids[i][0]] = theta_z


def cal_distance(id1, id2, pos):
    if id1 in pos and id2 in pos:
        distance = np.linalg.norm(pos[id1] - pos[id2])
        return distance
    else:
        return np.inf


def cal_angle(agent, vertex_id, next_id, pos):
    try:
        vertex = pos[vertex_id]
        next = pos[next_id]
        v1 = agent.position - vertex
        v2 = next - vertex
        cos_angle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(cos_angle) / np.pi * 180
        return angle
    except Exception:
        return np.inf


def set_as_maze(graph):
    verticies_1 = (0, 0, 1, 1, 2, 2, 3, 3, 4, 5,
                6, 7, 7, 10, 10, 11, 12, 13, 14, 17, 18, 18, 19)
    verticies_2 = (20, 21, 4, 20, 6, 11, 8, 16, 5,
                    10, 9, 8, 13, 11, 17, 14, 13, 15, 15, 18, 19, 21, 22)

    for vertex_1, vertex_2 in zip(verticies_1, verticies_2):
        graph.add_edge(vertex_1, vertex_2, cal_distance(vertex_1, vertex_2, position))
    return graph


def main():
    mtx, dist = init_parameters()
    agent = Agent(234, test=False)
    cap = open_camera()
    tick = 0
    agent_detect = False
    initialization = True

    while True:  
        frame = capture_frame(cap)
        corners, ids, _ = detect_aruco(frame)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(corners, 0.158, mtx, dist)
        # (rvec - tvec).any()
        
            for i in range(rvec.shape[0]):
                aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.1)
            aruco.drawDetectedMarkers(frame, corners, ids, (0, 0, 255))

            get_position(ids, tvec, position)
            get_orientation(ids, rvec, orientation)
            if agent.id in position and agent.id in orientation:
                agent.set_location()
                agent.set_orientation()
                agent_detect = True
            else:
                print("The agent is missing...")
            
            if initialization:
                if ids.shape[0] == 23:
                    maze = set_as_maze(Graph(23))
                    maze.find_path(15, 18)
                    initialization = False
                    print(maze.path)
                    agent.set_path(maze.path)
                else:
                    print('initializing maze...')

            if not initialization and agent_detect:
                if tick % 1 == 0:
                    agent.state_control()
                    print('14--{}--, 234--{}--'.format(position[14], position[agent.id]))
            
                tick += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            agent.stop()
            break
        cv2.imshow("Capture", frame)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()