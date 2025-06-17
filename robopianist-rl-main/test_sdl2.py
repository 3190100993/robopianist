import fluidsynth
# import sdl2
# import sdl2.sdlmixer
# import pygame

# # 初始化Pygame
# pygame.init()

# # 设置窗口尺寸
# screen = pygame.display.set_mode((640, 480))
# pygame.display.set_caption("SDL2 Test")

# # 主循环
# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     # 填充屏幕为白色
#     screen.fill((255, 255, 255))

#     # 更新显示
#     pygame.display.flip()

# # 退出Pygame
# pygame.quit()

# # 初始化SDL2
# sdl2.SDL_Init(sdl2.SDL_INIT_AUDIO)
# # 初始化SDL2_mixer (可选，用于更高级的音频控制)
# sdl2.sdlmixer.Mix_Init(sdl2.sdlmixer.MIX_INIT_OGG)
# sdl2.sdlmixer.Mix_OpenAudio(44100, sdl2.sdlmixer.MIX_DEFAULT_FORMAT, 2, 1024)
# 创建FluidSynth对象
fs = fluidsynth.Synth()

# 初始化SDL2音频驱动
result = fs.start(driver='pulseaudio')
print(result)

if result:
    print("音频驱动初始化成功！")
else:
    print("音频驱动初始化失败。")

# 关闭FluidSynth
fs.delete()
# sdl2.sdlmixer.Mix_CloseAudio()
# sdl2.sdlmixer.Mix_Quit()
# sdl2.SDL_Quit()
