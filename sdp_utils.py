def set_bitrate_hints(
    sdp: str,
    start_kbps: int,
    min_kbps: int,
    max_kbps: int,
    framerate: int | None = None,
    add_framerate_hint: bool = False,
) -> str:
    """
    öyle işte
    """
    lines = sdp.splitlines()
    out = []
    in_video = False
    inserted_b = False
    inserted_framerate = False

    for line in lines:
        if line.startswith("m="):
            in_video = line.startswith("m=video")
            inserted_b = False
            inserted_framerate = False

        out.append(line)

        if in_video and line.startswith("c=") and not inserted_b:
            out.append(f"b=AS:{max_kbps}")             # kbps
            out.append(f"b=TIAS:{max_kbps * 1000}")    # bps
            if add_framerate_hint and framerate and not inserted_framerate:
                out.append(f"a=framerate:{framerate}")
                inserted_framerate = True
            inserted_b = True
        
        # x-google ne hiçbir fikrim yok
        if in_video and line.startswith("a=fmtp:"):
            if ("x-google-max-bitrate" not in line and
                "x-google-min-bitrate" not in line and
                "x-google-start-bitrate" not in line):
                sep = ";" if ";" in line else " "
                out[-1] = (
                    f"{line}{sep}"
                    f"x-google-start-bitrate={start_kbps};"
                    f"x-google-min-bitrate={min_kbps};"
                    f"x-google-max-bitrate={max_kbps}"
                )

    return "\r\n".join(out) + "\r\n"
