package com.projeto.tcc.controllers;

import com.projeto.tcc.services.FrameService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.File;

@RestController
@RequestMapping(path = "frames")
public class FramesController {

    @Autowired
    private FrameService frameService;

    @PostMapping
    public File framesampleFile(
            @RequestBody File framesfile

    ) {
        return frameService.samplingFrames(framesfile);
    }
}
