from asyncio import sleep
from pathlib import Path

import imageio as iio
from tqdm import tqdm

from vuer import Vuer
from vuer.events import ClientEvent
from vuer.schemas import Scene, ImageBackground, SceneBackground

#import sys
#sys.path.append("/home/beantown/ran/mit/Scaffold-GS")
import render_ran as GS
init_camera = {'camera': {
    'matrix': [1.0, 0.0, 0.0, 0.0,
               0.0, 1.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0,
               0.0, 0.0, 0.0, 1.0],
    'position': [0.0, 0.0, 0.0],
    'fov': 75,
    },
}
others = GS.run_render() #load GS model
image = GS.run_render_with_view(init_camera, others) #get initial image

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
    session.set @ Scene(up=[0, -1, 0])

    while True:

        # 'jpg' encoding should give you about 30fps with a 16ms wait in-between.
        await sleep(1)


async def on_camera(event: ClientEvent, session):
    global image, others
    image = GS.run_render_with_view(event.value, others)
    assert event == "CAMERA_MOVE", "the event type should be correct"
    # print("camera event", event.etype, event.value)
    print("world transformation")
    world = event.value['world']
    print("position", world['position'])
    print("rotation", world['rotation'])

    session.upsert(
        ImageBackground(
            # Can scale the images down.
            image[::3, ::3],
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
    await sleep(0.0)

app.add_handler("CAMERA_MOVE", on_camera)
app.run()