#-*-coding:utf-8-*-

import sys
import re

from PyQt4 import QtCore, QtGui, QtNetwork

import errno

import pyqtgraph as pg
import numpy as np
from numpy.fft import rfft
#--------------------------------------------DEFINES------------------------------------------


#CAS
SERVER_HOST_CAS = '172.16.1.110'
SERVER_PORT_CAS = 20041


#HEMERA
SERVER_HOST = 'hemera'
SERVER_PORT = 10004

BUFFER_SIZE = 128

order_timing = 'name:Timing|method:subscribe\n'
order_timing = order_timing.encode()

order_voltage = 'name:VEPP/RF/U|method:subscribe\n'
order_voltage = order_voltage.encode()

order_energy = 'name:VEPP/Energy/Energy_NMR|method:subscribe\n'
order_energy = order_energy.encode()

order_regime = 'name:Regime|method:subscribe\n'
order_regime = order_regime.encode()

#---------------------------------------------------------------------------------------------

class MyWindow(QtGui.QWidget):

    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)
        
        self.pkt_head_delivered = 0
        self.currentFortune = b''
        self.len_to_receive = 0
        self.injection = 0
        self.bpm_number = 34
        
        self.ip = 0.
        self.ie = 0.

        self.energy = [350.]
        self.voltage = [13.]
        self.regime = 1.0
        self.reg = 1.0
        self.nus = 0.002
        
        self.phase = [0.0]
        self.nu_meas = [0.0]
        self.amplitude = [0.0]
        self.cos = [0.0]
        self.sin = [0.0]
        
        self.d_list = [0.0]
        self.c_list = [0.0]


        self.vbox = QtGui.QVBoxLayout()
        
        self.btnQuit = QtGui.QPushButton("Rec data")      
        self.connect(self.btnQuit, QtCore.SIGNAL("clicked()"), self.rec_stat)

        self.label = QtGui.QLabel('')
        self.label1 = QtGui.QLabel('')
        self.label2 = QtGui.QLabel('')

        self.font = QtGui.QFont()
        self.font.setBold(True)
        self.font.setPointSize(30)

        self.label1.setFont(self.font)
        self.label2.setFont(self.font)

                
        self.plot_space1 = pg.PlotWidget()
        self.plot_space2 = pg.PlotWidget()

#        self.vbox.addWidget(self.label)        
        self.vbox.addWidget(self.label1)        
        self.vbox.addWidget(self.label2)        
        self.vbox.addWidget(self.plot_space1)
        self.vbox.addWidget(self.plot_space2)
        self.vbox.addWidget(self.btnQuit)        
        self.setLayout(self.vbox)
        
        self.tcpSocket = QtNetwork.QTcpSocket(self)        
        self.tcpSocket.connectToHost(SERVER_HOST,SERVER_PORT)

        self.tcpSocket.readyRead.connect(self.readFortune)    
        self.connect(self.tcpSocket, QtCore.SIGNAL("connected()"), self.print_connected)
        self.connect(self.tcpSocket, QtCore.SIGNAL("disconnected()"), self.print_disconnected)
        self.tcpSocket.error.connect(self.print_error)

        self.timer = QtCore.QTimer()
        self.timer.connect(self.timer, QtCore.SIGNAL('timeout()'), self.reconnect)
       
        
        self.connect(self, QtCore.SIGNAL("external()"), self.beamhistory)
        self.connect(self, QtCore.SIGNAL("history_recorded()"), self.fft_naff)

#        self.casSocket = QtNetwork.QTcpSocket(self) 
#        self.casSocket.connectToHost(SERVER_HOST_CAS,SERVER_PORT_CAS)
	
#        self.connect(self.casSocket, QtCore.SIGNAL("connected()"), self.print_connected_cas)
#        self.connect(self.casSocket, QtCore.SIGNAL("disconnected()"), self.print_disconnected_cas)
#        self.casSocket.error.connect(self.print_error_cas)
#        self.casSocket.readyRead.connect(self.catch)
           



        self.infoSocket = QtNetwork.QTcpSocket(self)        
        self.infoSocket.connectToHost(SERVER_HOST_CAS,SERVER_PORT_CAS)
	
        self.connect(self.infoSocket, QtCore.SIGNAL("connected()"), self.print_connected_info)
        self.connect(self.infoSocket, QtCore.SIGNAL("disconnected()"), self.print_disconnected_cas)
        self.infoSocket.error.connect(self.print_error_cas)
        self.infoSocket.readyRead.connect(self.catch_info)
           
        self.casTimer = QtCore.QTimer()
        self.casTimer.connect(self.casTimer, QtCore.SIGNAL('timeout()'), self.reconnect_cas)

                   
#-------------------------------HEMERA------------------------------------------       			
        
    def print_connected(self):
        print('connection with HEMERA established')

    def print_disconnected(self):
        print('disconnected from HEMERA server')
        self.tcpSocket.abort()
        self.timer.start(1000)
                
    def print_error(self):
        print('Error with HEMERA occured')
        self.tcpSocket.abort()
        print('HEMERA socket is closed')
        self.timer.start(1000)

    def reconnect(self):
        print('Trying to solve the problem with HEMERA...')        
        try:
            self.tcpSocket.connectToHost(SERVER_HOST,SERVER_PORT)
            if self.tcpSocket.waitForConnected() == True:
                print('Reconnected with HEMERA')
                self.timer.stop()
            else:
                print('Connection with HEMERA failed')    
        except:
            print('exeption occured, hemera')  

#----------------------------CAS-------------------------------------------------       			
                     
    def print_connected_cas(self):
        print('connection with CAS established')
        self.casSocket.writeData(order_timing)

    def print_connected_info(self):
        print('connection with CAS established')
        self.casSocket.writeData(order_timing)	
        self.infoSocket.writeData(order_regime)
        self.infoSocket.writeData(order_energy)
        self.infoSocket.writeData(order_voltage)
        self.read_IE()


    def print_disconnected_cas(self):
        print('disconnected from CAS server')
#        self.casSocket.abort()
        self.infoSocket.abort()
        self.casTimer.start(1000)
                
    def print_error_cas(self):
        print('Error occured with CAS server')
#        self.casSocket.abort()
        self.infoSocket.abort()
        print('CAS socket is closed')
        self.casTimer.start(1000)

    def reconnect_cas(self):
        print('Trying to solve the problem with CAS...')        
        try:
#            self.casSocket.connectToHost(SERVER_HOST_CAS,SERVER_PORT_CAS)
            self.infoSocket.connectToHost(SERVER_HOST_CAS,SERVER_PORT_CAS)
            if self.casSocket.waitForConnected() == True:
                print('Reconnected with CAS')
                self.casTimer.stop()
            else:
                print('Connection with CAS failed')    
        except:
            print('exeption')             


#    def catch(self):
#        try:
#            data = self.casSocket.read(BUFFER_SIZE)
#        except:
#            print('Exeption occured ')
        
#        if data:
#            self.catch_sollution(data)  
#        else:
#            print('Nothing appeared')


#    def catch_sollution(self, data1):
#        data1 = data1.decode()
#        data1 = data1.strip()       
#        data1 = data1.split('\n')

#        for elem in data1:
#            elem = elem.split('|')        
#            if elem[-1] == 'val:Wypusk':
#                print(elem)
#                self.injection = 1
#                self.regime = self.reg                
#                print(self.regime)            




    def catch_info(self):
        try:
            data = self.infoSocket.read(BUFFER_SIZE)
        except:
            print('Exeption durin CAS reading')
        
        if data:
            self.data_collect(data)  
        else:
            print('No data appeared')

    def data_collect(self, array):

        try:
            array = array.decode()
            array = array.strip()
            array = array.split('\n')
                  		
            for elem in array:
                elem = elem.split('|')
		
                if elem[-1] == 'val:Wypusk':
                    print(elem)
                    self.injection = 1
                    self.regime = self.reg                
                    print(self.regime)            
0		
                if elem[0] == 'name:VEPP/Energy/Energy_NMR':
                    E = elem[2]
                    E = E.split(':')
                    self.energy.append(float(E[-1]))
                if elem[0] == 'name:VEPP/RF/U':
                    U = elem[2]
                    U = U.split(':')
                    self.voltage.append(float(U[-1]))
                if elem[0] == 'name:Regime':
                    R = elem[-1]
                    R = R.split(':')
                    self.reg = float(R[-1])
            
        except:
            print('INFO EXCEPTION')     		
              
#--------------------------------------------------------------------------------       			
    def readHead(self):
        try:
            data = self.tcpSocket.read(BUFFER_SIZE)
        except:
            print('error occured during reading head')
            data = b''
            
        return data

    def readFortune(self):
		
        if self.pkt_head_delivered == 0:
            buf = self.readHead()
            try:
                self.len_to_receive, self.name, self.mode = self.parse_pkt_head(buf)
                self.pkt_head_delivered = 1
            except:
                print('error occured during parsing')
            
        elif self.pkt_head_delivered == 1:
            try:
                data = self.tcpSocket.read(self.len_to_receive)
            except:
                print('error occured during reading package')    

            if (self.len_to_receive - len(data)) == 0:
                self.currentFortune += data
                
                
                self.emit(QtCore.SIGNAL("data_appeared()"))
                if self.mode == 'ext':
                    self.emit(QtCore.SIGNAL("external()"))
                if self.mode == 'int':
                    self.emit(QtCore.SIGNAL("internal()"))                    
                    
                self.pkt_head_delivered = 0
#                if self.mode == 'ext':
#                    print('data delivered [full = %d bytes]' % len(data))
                self.currentFortune = b''      												

            else:
                self.currentFortune += data      												
                self.len_to_receive -= len(data)
#                if self.mode == 'ext':
#                    print('data delivered [part = %d bytes]' % len(data))      												


    def parse_pkt_head(self,buf):
        data = buf.decode('ascii')
        data = data.strip()
        data = re.split(r'[:| ]+', data)
        data_size = int(data[10])
        bpm_name = data[1]
        mode = data[8]
        if mode == 'ext':
            print(data)
        return [data_size, bpm_name, mode]

    def get_position(self):
        dt  = np.dtype('float32')
        pkt = np.frombuffer(self.currentFortune, dtype=dt)
        pkt = np.split(pkt, 4)
        return pkt

#------------------------------------------------------------------------------------------

                
    def beamhistory(self):

        i=0    			                     
        while self.ebep[i] < self.energy[-1]:
            i += 1
            
        M = i
        self.kbep = 1000*(self.ibep[M] - self.ibep[M-1])/(self.ebep[M] - self.ebep[M-1])
# It isnt global, because VEPP Energy might be changed during running
            	
        if 	self.injection == 1:	

            print(self.regime)
            
            if self.name == 'pic03':
                package = self.get_position()                       
                self.x3 = package[1]
                
                self.delta = np.max(package[3][6000:8000]) - np.mean(package[3][0:28])
                self.curr = np.mean(package[3][0:28])
                
                self.D = 670.
                         
                if self.regime == 4:
                    self.x = self.x3
                    print('p inj')
                    self.bpm_number = 3
                    self.emit(QtCore.SIGNAL("history_recorded()"))
                    self.injection = 0
                    
                    

            else:
                package = self.get_position()                       
                self.x4 = package[1]

                self.delta = np.max(package[3][6000:8000]) - np.mean(package[3][0:28])
                self.curr = np.mean(package[3][0:28])
                
                self.D = 720.

                if self.regime == 2:
                    self.x = self.x4
                    print('e inj')
                    self.bpm_number = 4
                    self.emit(QtCore.SIGNAL("history_recorded()"))
                    self.injection = 0
                                             



    def fft_naff(self):

        self.zero = np.mean(self.x)

        self.plot_space1.clear() 
        self.x = (self.x - np.mean(self.x))*((self.delta + self.curr)/self.delta)
        self.x = self.x - np.mean(self.x)

        self.curve1 = self.plot_space1.plot(self.x + self.zero, pen=(0,0,200))           

        self.d_list.append(self.delta)
        self.c_list.append(self.curr) 

        y = rfft(self.x)/len(self.x)
        Q = np.linspace(0.,0.5, len(y))

        nu_cas = np.sqrt(14*0.036*(self.voltage[-1])/(2*np.pi*(self.energy[-1])*1000))

        
        if nu_cas - nu_cas*0.3 > 0:
            nu_start = nu_cas - nu_cas*0.3
        else:
            nu_start = 0
        nu_stop = nu_cas + nu_cas*0.3
        

        i = int(len(self.x)*nu_start)
        f = int(len(self.x)*nu_stop)
        
        
        y1 = y[i:f]
        
        p = np.argmax(np.abs(y1)) + i
        nu_max = Q[p]
        
        if p>5:
            pi = p-5
        else:
            pi = 0
        pf = p+5

        self.plot_space2.clear()                 
        self.curve2 = self.plot_space2.plot(Q[pi:pf],abs(y[pi:pf]), pen=(0,0,200))
        self.curve3 = self.plot_space2.plot([nu_max, nu_max], [0,max(abs(y1))], pen=(0,0,200))

        nu_start = Q[p-1]
        nu_stop = Q[p+1]
        delta_nu = 10**(-6)
        nu = np.linspace(nu_start, nu_stop, int(np.abs(nu_stop - nu_start)/delta_nu))

        n = np.arange(0,len(self.x))
        naff_list = list()

        for a in nu:
            naff_list.append(np.dot(self.x, np.exp(2*np.pi*1j*n*a)))   

        y =  np.abs(np.sqrt(2.0)*np.array(naff_list)/len(self.x)) 
       
        p_naff = np.argmax(y)
        nu_naff = nu[p_naff]
        self.nu_meas.append(nu_naff)

        self.curve4 = self.plot_space2.plot(nu, y, pen=(200,0,0))
        self.curve5 = self.plot_space2.plot([nu_naff, nu_naff], [0,max(y)], pen=(200,0,0))   

        cos = np.sqrt(2.0)*np.cos(2*np.pi*nu_naff*n)
        sin = -np.sqrt(2.0)*np.sin(2*np.pi*nu_naff*n)
        A1 = (1/len(self.x))*np.dot(self.x, cos)
        A2 = (1/len(self.x))*np.dot(self.x, sin)
        A = np.sqrt(2.0)*np.sqrt(A1**2 + A2**2)
        
        self.amplitude.append(A)
        
        x_out = A1*cos + A2*sin + self.zero
              
        phi = np.arctan2(A2,A1)
        dk = -14*0.036*A2/(self.D*2*np.pi*nu_naff)
        dI = np.sign(A1)*self.energy[-1]*(A/self.D)*self.kbep
        
        x_x = A*np.cos(2*np.pi*nu_naff*n + phi)  + self.zero

        self.phase.append(phi)
        
        self.label1.setText('A1= %.4f A2 = %.4f'%(A1, A2))
        self.label2.setText('dK = %.3f ,dI = %.3f'%(dk, dI))
        
        self.cos.append(A1)
        self.sin.append(A2)        
        
        self.curve6 = self.plot_space1.plot(x_x, pen=(0,200,0))      


    def rec_stat(self):
        stat = open('statistic.txt', 'w')

        i = 0	
        stat.write('nu_pi phase ampl cos sin current delta'+'\n')
	if len(self.nu_meas) > 0:
            for i in np.arange(len(self.nu_meas)):	
                stat.write('%f %f %f %f %f %f %f \n'%(self.nu_meas[i], self.phase[i], self.amplitude[i], self.cos[i], self.sin[i], self.c_list[i], self.d_list[i]))
        else:
            print('nothing to save, sorry...')
	
	stat.close()
            
            
    def read_IE(self):
        ie = open('/home/users/work/Rabusov/Histories/IEBBmeas.dat')
        self.ibep = list()
        self.ebep = list()
        i=0
        for line in ie:
            if i != 0:
                line = line.strip()
                line = line.split()
                self.ibep.append(float(line[0]))
                self.ebep.append(float(line[1]))
            i += 1
        ie.close()


if __name__ == "__main__":

    app = QtGui.QApplication(sys.argv)

    window = MyWindow()
    window.setWindowTitle('INJECTION')
    window.resize(800, 600)
    
    window.show()

    sys.exit(app.exec_())
