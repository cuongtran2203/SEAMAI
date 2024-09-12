# Communicator Object

import pickle, time
import struct
import socket

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Communicator(object):
	def __init__(self, index, recv_timeout = 5, buffer_size =4, bandwidth = 1000):
		self.recv_timeout = recv_timeout
		self.buffer_size = buffer_size
		self.index = index
		self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
		self.bw = bandwidth #MBs
	def send_msg(self, sock, msg):
		try:
			msg_pickle = pickle.dumps(msg)
			sock.sendall(struct.pack(">I", len(msg_pickle)))
			sock.sendall(msg_pickle)
			# logger.info(msg[0] + ' sent to ' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))
		except BaseException as e:
			# logger.info('Error:' + str(e))
			pass

	def recv_msg(self, sock, expect_msg_type=None):
		t0=time.time()
		while time.time()-t0<self.recv_timeout:
			try:
				msg_len = struct.unpack(">I", sock.recv(self.buffer_size))[0]
				msg = sock.recv(msg_len, socket.MSG_WAITALL)
				msg = pickle.loads(msg)


				if type(self.bw) is not dict:	time.sleep(float(msg_len)/((self.bw/8)*1_000_000))
				elif self.bw[str(sock.getpeername()[0])] == float('inf'): pass
				else: time.sleep(float(msg_len) / ((self.bw[str(sock.getpeername()[0])] / 8) * 1_000_000))

				# logger.info(msg[0] + ' received from ' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))
				if expect_msg_type is not None:
					if msg[0] == expect_msg_type:
						return msg
					elif msg[0] != expect_msg_type:
						# logger.info("Expected " + expect_msg_type + " but received " + msg[0])
						return [expect_msg_type,None]
			except BaseException as e:
				continue
		else:
			# logger.info('socket.timeout')
			return [expect_msg_type,None]


