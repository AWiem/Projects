package com.pets.pets_service.Registration;

import lombok.AllArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping(path = "api/v1/ppregistration")
@AllArgsConstructor
public class PPRegistrationController {

    private final PPRegistrationService registrationService;

    @PostMapping
    public String register(@RequestBody PPRegistrationRequest request) {
        return "{\"token\":\""+registrationService.register(request)+"\"}";
    }

    @GetMapping(path = "confirm")
    public String confirm(@RequestParam("token") String token) {
        return registrationService.confirmToken(token);
    }

}