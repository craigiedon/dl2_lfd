class AnimType(Enum):
    
    JS = 0,
    HTML5 = 1,
    MP4 = 2,


def attribution_video(model, demo_dataset):
    full_model.eval()
    # Split out data
    with torch.no_grad():
        raw_ims = [d["raw_image"] for d in demo_dataset]
        eval_ims = torch.stack([d["image"] for d in demo_dataset]).to(device)
        eval_poses = torch.stack([d["pose"] for d in demo_dataset]).to(device)
        eval_controls_true = torch.stack([d["control"] for d in demo_dataset]).numpy()
        eval_controls_est = model(eval_ims, eval_poses)["output"]

    # Setup subplots and axes
    fig = plt.figure(figsize=(15,5))
    img_heatmap_ax = fig.add_subplot(1,2,1)
    control_ax = fig.add_subplot(2,2,2)
    control_est_ax = fig.add_subplot(2,2,4)

    print(len(demo_dataset))
    control_ax.set_xlim(0, len(demo_dataset))
    control_ax.set_ylim(-np.pi / 2.0, np.pi / 2.0)

    control_est_ax.set_xlim(0, len(demo_dataset))
    control_est_ax.set_ylim(-np.pi / 2.0, np.pi / 2.0)

    # Initial drawings for each axis
    im_canvas = img_heatmap_ax.imshow(raw_ims[0])
    true_canvases = control_ax.plot(eval_controls_true)
    est_canvases = control_est_ax.plot(eval_controls_est.cpu().detach().numpy())

    # Initial overlays for end-effector position and velocity computations
    end_eff_circle = img_heatmap_ax.add_patch(patches.Circle((0,0), 3, color="g"))
    img_heatmap_ax.add_patch(patches.FancyArrowPatch((50,50), (75,75), color="blue", mutation_scale=4))
    img_heatmap_ax.add_patch(patches.FancyArrowPatch((50,50), (75,75), color="red", mutation_scale=4))

    moved_poses = eval_poses.cpu().numpy() + eval_controls_true
    moved_poses_est = eval_poses.cpu().numpy() + eval_controls_est.cpu().detach().numpy()

    def anim_step(i):
        ee_pos = robot_model.project_camera(eval_poses[i])
        ee_vel = robot_model.project_camera(moved_poses[i])
        ee_vel_est = robot_model.project_camera(moved_poses_est[i])
        end_eff_circle.center = (ee_pos[0],ee_pos[1])

        del img_heatmap_ax.patches[:]
        img_heatmap_ax.add_patch(
            patches.FancyArrowPatch((ee_pos[0], ee_pos[1]), (ee_vel[0], ee_vel[1]), color="b", mutation_scale=4))

        img_heatmap_ax.add_patch(
            patches.FancyArrowPatch((ee_pos[0], ee_pos[1]), (ee_vel_est[0], ee_vel_est[1]), color="orange", mutation_scale=4))

        im_canvas.set_data(raw_ims[i])
        for j, true_canvas in enumerate(true_canvases):
            true_canvas.set_data(range(i), eval_controls_true[:i, j])
            est_canvases[j].set_data(range(i), eval_controls_est[:i, j].cpu().detach().numpy())
        return [im_canvas] + true_canvases + est_canvases

    ani = animation.FuncAnimation(fig, anim_step, interval=200, frames=len(demo_dataset), blit=True, repeat=False)
    plt.show()
    return ani

def display_animation(animation, display_mode):
    if display_mode is AnimType.JS:
        print("JS display!")
        from IPython.display import HTML
        return HTML(animation.to_jshtml())
    if display_mode is AnimType.HTML5:
        from IPython.display import HTML
        return HTML(animation.to_html5_video())
    if display_mode is AnimType.MP4:
        animation.save("anim.mp4")

