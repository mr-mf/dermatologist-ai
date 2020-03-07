from aiohttp import web


class Controller:
    __app__ = web.Application()

    @classmethod
    async def run_app(cls):
        """
        Route the requests to the methods that can handle them
        :return:  __app__
        """
        cls.__app__.add_routes([web.get("/", cls.handle)])
        return cls.__app__

    async def handle(request):
        """
        Accept the photo, parse into acceptable format and make a prediction
        :return: json
        """
        return web.Response(text="Hello world")


if __name__ == '__main__':
    web.run_app(Controller.run_app())
