from aiohttp import web
from predictor import Predictor


class Controller:
    __app = web.Application()
    __predictor = Predictor()

    @classmethod
    async def run_app(cls):
        """
        Route the requests to the methods that can handle them
        :return:  __app
        """
        cls.__app.add_routes([web.post("/predict", cls.handle)])
        return cls.__app

    @classmethod
    async def handle(cls, request: web.Request):
        """
        Accept the photo, parse into acceptable format and make a prediction
        :return: json
        """
        post = await request.post()
        image = post.get('image')
        if image:
            img_content = image.file.read()
        output = await cls.__predictor.predict(img_content)
        return web.json_response({"melanoma probability": str(output)})


if __name__ == '__main__':
    web.run_app(Controller.run_app())
