package com.projeto.tcc.services;

import org.springframework.stereotype.Service;

import java.io.File;

@Service
public class FrameService {

    public File samplingFrames(File framesFile){
        return new File(String.valueOf(framesFile));
    }
}
