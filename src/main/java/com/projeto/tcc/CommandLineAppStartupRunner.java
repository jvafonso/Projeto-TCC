package com.projeto.tcc;

import com.projeto.tcc.services.FrameService2;
import com.projeto.tcc.services.MenuService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class CommandLineAppStartupRunner implements CommandLineRunner {

    @Autowired
    private FrameService2 frameService2;

    @Autowired
    private MenuService menuService;

    @Override
    public void run(String... args) throws Exception {
        menuService.startMenu();
        System.exit(0);
    }
}
