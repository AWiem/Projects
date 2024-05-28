package com.pets.pets_service.Registration;

import lombok.AllArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping(path = "api/v1/vetregistration")
@AllArgsConstructor
public class VetRegistrationController {

    private final VetRegistrationService registrationService;

    @PostMapping
    public String register(@RequestBody VetRegistrationRequest request) {
        return "{\"token\":\""+registrationService.register(request)+"\"}";
    }

    @GetMapping(path = "confirm")
    public String confirm(@RequestParam("token") String token) {
        return registrationService.confirmToken(token);
    }

}