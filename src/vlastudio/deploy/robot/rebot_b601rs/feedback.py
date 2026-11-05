"""Receive RobStride active reports with SocketCAN kernel arrival timestamps."""
import math
import socket
import struct
import time


class ActiveReports:
    def __init__(self, channel):
        self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        # Linux aarch64 SO_TIMESTAMPNS uses a pair of native longs.
        self.timestamp_option = getattr(socket, 'SO_TIMESTAMPNS', 35)
        try:
            # CAN_ERR_FLAG in a Linux filter mask selects the error-frame list.
            mask = socket.CAN_EFF_FLAG | socket.CAN_RTR_FLAG | 0x1f00ffff
            filters = b''.join(struct.pack('=II', 0x980000fd | (mid << 8), mask)
                               for mid in range(1, 8))
            self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
            self.sock.setsockopt(socket.SOL_SOCKET, self.timestamp_option, 1)
            self.sock.bind((channel,))
            self.sock.setblocking(False)
        except BaseException:
            self.close()
            raise

    def drain(self):
        offset = time.monotonic() - time.time()
        # Bounded even if a malfunctioning motor floods the bus.
        for _ in range(512):
            try:
                frame, ancillary, flags, _ = self.sock.recvmsg(16, socket.CMSG_SPACE(16))
            except BlockingIOError:
                return
            cid, dlc, data = struct.unpack('=IB3x8s', frame)
            if dlc != 8 or flags & socket.MSG_CTRUNC:
                raise RuntimeError('Incomplete CAN active report')
            for level, kind, value in ancillary:
                if level == socket.SOL_SOCKET and kind == self.timestamp_option:
                    sec, ns = struct.unpack('@ll', value)
                    stamp = sec + ns * 1e-9 + offset
                    break
            else:
                raise RuntimeError('Missing CAN receive timestamp')
            raw = int.from_bytes(data[:2], 'big')
            # MotorBridge RS00/RS06 status decoder: neutral code 0x7fff, +/-4 pi rad.
            position = (raw / 32767. - 1.) * (4 * math.pi)
            yield ((cid >> 8) & 255) - 1, stamp, position, (cid >> 22) & 3, (cid >> 16) & 63

    def close(self):
        self.sock.close()
