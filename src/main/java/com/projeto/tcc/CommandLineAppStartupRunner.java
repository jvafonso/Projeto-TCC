package com.projeto.tcc;

import com.projeto.tcc.services.FrameService;
import com.projeto.tcc.services.FrameService2;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class CommandLineAppStartupRunner implements CommandLineRunner {
    @Autowired
    private FrameService frameService;

    @Autowired
    private FrameService2 frameService2;

    @Override
    public void run(String... args) throws Exception {
        frameService2.samplingFrames();
    }
}
