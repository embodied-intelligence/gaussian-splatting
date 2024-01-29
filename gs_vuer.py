import time
from asyncio import sleep
from pathlib import Path
from pprint import pprint

from params_proto import ParamsProto
from vuer import Vuer
from vuer.events import ClientEvent
from vuer.schemas import Scene, ImageBackground

# import sys
# sys.path.append("/home/beantown/ran/mit/Scaffold-GS")
import render_ran as GS

class GSVuer(ParamsProto):
    eps = 0.001
    fps = 30

others = GS.main()  # load GS model
timestamp = time.time()
render_params = None

assets_folder = Path(__file__).parent / "../../assets"
app = Vuer(
    queries=dict(
        reconnect=True,
        grid=False,
        backgroundColor="black",
        # port=7010,
        # uri="ws://localhost:7010",
    ),
    static_root=assets_folder,
    # debug=True,
)


@app.spawn
async def show_heatmap(session):
    global render_params
    session.set @ Scene(up=[0, -1, 0]) # 0,0,-1

    while True:
        if render_params is None:
            await sleep(0.016)
            continue

        image = GS.run_render_with_view(render_params, others)
        render_params = None

        session.upsert(
            ImageBackground(
                # Can scale the images down.
                # image,
                #image[::2, ::2],
                image,
                # One of ['b64png', 'png', 'b64jpg', 'jpg']
                # 'b64png' does not work for some reason, but works for the nerf demo.
                # 'jpg' encoding is significantly faster than 'png'.
                format="jpg",
                quality="90",
                key="background",
                interpolate=True,
            ),
            to="bgChildren",
        )
        # 'jpg' encoding should give you about 30fps with a 16ms wait in-between.
        await sleep(0.030)


async def on_camera(event: ClientEvent, session):
    global render_params, timestamp
    assert event == "CAMERA_MOVE", "the event type should be correct"

    render_params = event.value
    timestamp = time.time()


app.add_handler("CAMERA_MOVE", on_camera)
app.run()
