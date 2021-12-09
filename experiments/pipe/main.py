# hivemind/dht/__init__.py
import multiprocessing as mp
import asyncio
import uvloop


def switch_to_uvloop() -> asyncio.AbstractEventLoop:
    """stop any running event loops; install uvloop; then create, set and return a new event loop"""
    try:
        asyncio.get_event_loop().stop()  # if we're in jupyter, get rid of its built-in event loop
    except RuntimeError as error_no_event_loop:
        pass  # this allows running DHT from background threads with no event loop
    uvloop.install()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


class TestPipe(mp.Process):
    def __init__(self):
        # 这一句忘记了的话就不能执行!
        super().__init__()
        self._inner_pipe, self._outer_pipe = mp.Pipe(duplex=True)
        self.start()

    def send_msg(self):
        print("sending")
        self._outer_pipe.send(("_shutdown", ["args"], {"kwargs_0": 0}))
        print("sent")

    def run(self) -> None:
        loop = switch_to_uvloop()
        # pipe_semaphore = asyncio.Semaphore(value=0)
        # loop.add_reader(self._inner_pipe.fileno(), pipe_semaphore.release)

        async def _run():
            # self.recv_msg()
            """Start automatically"""
            print("start recving")

            while True:
                method, args, kwargs = self._inner_pipe.recv()
                if method is not None:
                    print(method, args, kwargs)

        loop.run_until_complete(_run())

    # def recv_msg(self):
    #     """Start automatically"""
    #     print("start recving")

    #     while True:
    #         method, args, kwargs = self._inner_pipe.recv()
    #         if method is not None:
    #             print(method, args, kwargs)


test_pipe = TestPipe()
test_pipe.send_msg()

print("main done!")
