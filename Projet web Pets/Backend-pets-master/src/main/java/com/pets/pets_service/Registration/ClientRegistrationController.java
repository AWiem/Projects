package com.pets.pets_service.Registration;

import lombok.AllArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping(path = "api/v1/registration")
@AllArgsConstructor
public class ClientRegistrationController {

    private final ClientRegistrationService registrationService;

    @PostMapping
    public String register(@RequestBody ClientRegistrationRequest request) {
        return "{\"token\":\""+registrationService.register(request)+"\"}";
    }

    @GetMapping(path = "confirm")
    public String confirm(@RequestParam("token") String token) {
        return registrationService.confirmToken(token);
    }

}